import difflib
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import sentry_sdk
import torch
from pydantic import BaseModel, ValidationInfo, field_validator
from sentence_transformers import SentenceTransformer

from seer.db import DbGroupingRecord, Session

logger = logging.getLogger("grouping")


class GroupingRequest(BaseModel):
    project_id: int
    stacktrace: str
    message: str
    group_id: int | None = None
    stacktrace_hash: str | None = None
    k: int = 1
    threshold: float = 0.01

    @field_validator("stacktrace", "message")
    @classmethod
    def check_field_is_not_empty(cls, v, info: ValidationInfo):
        if not v:
            raise ValueError(f"{info.field_name} must be provided and not empty.")
        return v


class GroupingRecord(BaseModel):
    group_id: int | None
    project_id: int
    message: str
    stacktrace_embedding: np.ndarray
    stacktrace_hash: str | None

    def to_db_model(self) -> DbGroupingRecord:
        return DbGroupingRecord(
            group_id=self.group_id,
            project_id=self.project_id,
            message=self.message,
            stacktrace_embedding=self.stacktrace_embedding,
            stacktrace_hash=self.stacktrace_hash,
        )

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda x: x.tolist()  # Convert ndarray to list for serialization
        }


class GroupingResponse(BaseModel):
    parent_group_id: Optional[int]
    stacktrace_distance: float
    message_distance: float
    should_group: bool


class SimilarityResponse(BaseModel):
    responses: List[GroupingResponse]
    token: Optional[int]


class GroupingLookup:
    """Manages the grouping of similar stack traces using sentence embeddings and pgvector for similarity search.

    Attributes:
        model (SentenceTransformer): The sentence transformer model for encoding text.

    """

    def __init__(self, model_path: str, data_path: str):
        """
        Initializes the GroupingLookup with the sentence transformer model.

        :param model_path: Path to the sentence transformer model.
        """
        model_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = SentenceTransformer(
            model_path,
            trust_remote_code=True,
            device=model_device,
        )
        logger.info(f"GroupingLookup model initialized using device: {model_device}")
        sentry_sdk.capture_message(f"GroupingLookup model initialized using device: {model_device}")
        self.initialize_db(data_path)

    def initialize_db(self, data_path: str):
        """
        Initializes the database with records from a pickle file if a specific record does not exist.

        This method checks for the existence of a record with a specific group_id in the database.
        If the record exists, the database is assumed to be initialized, and the method returns early.
        If the record does not exist, the method proceeds to load data from a pickle file located at
        `data_path` and populates the database with these records.

        :param data_path: The file path to the pickle file containing the records to load into the database.
        """
        with Session() as session:
            key_group_id = 4506283937  # TODO: less hacky solution to populating the DB if needed
            record_exists = (
                session.query(DbGroupingRecord)
                .filter(DbGroupingRecord.group_id == key_group_id)
                .first()
                is not None
            )

            if record_exists:
                return

            with open(data_path, mode="rb") as records_file:
                records_df = pd.read_pickle(records_file)
                for _, row in records_df.iterrows():
                    new_record = DbGroupingRecord(
                        group_id=row["group_id"],
                        project_id=row["project_id"],
                        message=row["message"],
                        stacktrace_embedding=row["stacktrace_embedding"].astype(np.float32),
                    )
                    session.add(new_record)
                session.commit()

    def encode_text(self, stacktrace: str) -> np.ndarray:
        """
        Encodes the stacktrace using the sentence transformer model.

        :param stacktrace: The stacktrace to encode.
        :return: The embedding of the stacktrace.
        """
        return self.model.encode(stacktrace)

    def get_nearest_neighbors(self, issue: GroupingRequest) -> SimilarityResponse:
        """
        Retrieves the k nearest neighbors for a stacktrace within the same project and determines if they should be grouped.
        If no records should be grouped, inserts the request as a new GroupingRecord into the database.

        :param issue: The issue containing the stacktrace, similarity threshold, and number of nearest neighbors to find (k).
        :return: A SimilarityResponse object containing a list of GroupingResponse objects with the nearest group IDs,
                 stacktrace similarity scores, message similarity scores, and grouping flags.
        """
        with Session() as session:
            # If an exact match of the stacktrace hash is found, return this record
            if hasattr(issue, "stacktrace_hash") and issue.stacktrace_hash:
                existing_record = (
                    session.query(DbGroupingRecord)
                    .filter_by(stacktrace_hash=issue.stacktrace_hash)
                    .first()
                )
                if existing_record:
                    similarity_response = SimilarityResponse(responses=[], token=None)
                    similarity_response.responses.append(
                        GroupingResponse(
                            parent_group_id=existing_record.group_id,
                            stacktrace_distance=0.00,
                            message_distance=0.00,
                            should_group=True,
                        )
                    )
                    return similarity_response

            embedding = self.encode_text(issue.stacktrace).astype("float32")

            results = (
                session.query(
                    DbGroupingRecord,
                    DbGroupingRecord.stacktrace_embedding.cosine_distance(embedding).label(
                        "distance"
                    ),
                )
                .filter(
                    DbGroupingRecord.project_id == issue.project_id,
                    DbGroupingRecord.stacktrace_embedding.cosine_distance(embedding) <= 0.15,
                    DbGroupingRecord.group_id != issue.group_id,
                    DbGroupingRecord.group_id != None,
                )
                .order_by("distance")
                .limit(issue.k)
                .all()
            )

            # If no existing groups within the threshold, insert the request as a new GroupingRecord
            token = None
            if not any(distance <= issue.threshold for _, distance in results):
                token = self.insert_new_grouping_record(session, issue, embedding)

            session.commit()

        similarity_response = SimilarityResponse(responses=[], token=token)
        for record, distance in results:
            message_similarity_score = difflib.SequenceMatcher(
                None, issue.message, record.message
            ).ratio()
            should_group = distance <= issue.threshold

            similarity_response.responses.append(
                GroupingResponse(
                    parent_group_id=record.group_id,
                    stacktrace_distance=distance,
                    message_distance=1.0 - message_similarity_score,
                    should_group=should_group,
                )
            )

        return similarity_response

    def insert_new_grouping_record(
        self, session, issue: GroupingRequest, embedding: np.ndarray
    ) -> int:
        """
        Inserts a new GroupingRecord into the database if the group_id does not already exist.
        If new grouping record was created, return the id.

        :param session: The database session.
        :param issue: The issue to insert as a new GroupingRecord.
        :param embedding: The embedding of the stacktrace.
        """
        existing_record = None
        if issue.group_id:
            existing_record = (
                session.query(DbGroupingRecord).filter_by(group_id=issue.group_id).first()
            )

        if existing_record is None:
            new_record = GroupingRecord(
                group_id=issue.group_id,
                project_id=issue.project_id,
                message=issue.message,
                stacktrace_embedding=embedding,
                stacktrace_hash=issue.stacktrace_hash,
            ).to_db_model()
            session.add(new_record)
            session.commit()
            return new_record.id

        return existing_record.id
