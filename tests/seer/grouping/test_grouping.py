import unittest

from seer.db import DbGroupingRecord, Session
from seer.grouping.grouping import GroupingRequest, GroupingResponse, SimilarityResponse
from seer.inference_models import grouping_lookup


class TestGrouping(unittest.TestCase):
    def test_get_nearest_neighbors_no_hash_has_neighbor(self):
        """
        Tests get_nearest_neighbors when the request has no stacktrace hash and a neighbor exists
        within the threshold.
        """
        with Session() as session:
            embedding = grouping_lookup().encode_text("stacktrace")
            grouping_request = GroupingRequest(
                group_id=1, project_id=1, stacktrace="stacktrace", message="message"
            )
            grouping_lookup().insert_new_grouping_record(session, grouping_request, embedding)
            session.commit()

        grouping_request = GroupingRequest(
            group_id=2, project_id=1, stacktrace=b"stacktrace", message="message"
        )

        response = grouping_lookup().get_nearest_neighbors(grouping_request)
        assert response == SimilarityResponse(
            responses=[
                GroupingResponse(
                    parent_group_id=1,
                    stacktrace_distance=0.0,
                    message_distance=0.0,
                    should_group=True,
                )
            ],
            token=None,
        )

    def test_get_nearest_neighbors_has_hash_has_neighbor(self):
        """
        Tests get_nearest_neighbors when the request has a stacktrace hash and this hash matches
        an existing record (ie. the neighbor exists within the threshold)
        """
        with Session() as session:
            embedding = grouping_lookup().encode_text("stacktrace")
            grouping_request = GroupingRequest(
                group_id=1,
                project_id=1,
                stacktrace="stacktrace",
                message="message",
                stacktrace_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            grouping_lookup().insert_new_grouping_record(session, grouping_request, embedding)
            session.commit()

        grouping_request = GroupingRequest(
            group_id=2,
            project_id=1,
            stacktrace=b"stacktrace",
            message="message",
            stacktrace_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            k=1,
            threshold=0.01,
        )

        response = grouping_lookup().get_nearest_neighbors(grouping_request)
        assert response == SimilarityResponse(
            responses=[
                GroupingResponse(
                    parent_group_id=1,
                    stacktrace_distance=0.0,
                    message_distance=0.0,
                    should_group=True,
                )
            ],
            token=None,
        )

    def test_get_nearest_neighbors_no_hash_no_neighbor(self):
        """
        Tests get_nearest_neighbors when the request has no stacktrace hash and no neighbor exists
        within the threshold. Assert that a new record was added and its id is returned.
        """
        grouping_request = GroupingRequest(
            group_id=2,
            project_id=1,
            stacktrace=b"stacktrace",
            message="message",
            k=1,
            threshold=0.01,
        )

        response = grouping_lookup().get_nearest_neighbors(grouping_request)
        with Session() as session:
            new_record = session.query(DbGroupingRecord).filter_by(group_id=2).first()

        assert new_record
        assert response == SimilarityResponse(responses=[], token=new_record.id)

    def test_get_nearest_neighbors_has_hash_no_neighbor(self):
        """
        Tests get_nearest_neighbors when the request has a stacktrace hash and no neighbor exists
        within the threshold. Assert that a new record was added and its id is returned.
        """
        grouping_request = GroupingRequest(
            group_id=2,
            project_id=1,
            stacktrace=b"stacktrace",
            message="message",
            stacktrace_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            k=1,
            threshold=0.01,
        )

        response = grouping_lookup().get_nearest_neighbors(grouping_request)
        with Session() as session:
            new_record = session.query(DbGroupingRecord).filter_by(group_id=2).first()

        assert new_record
        assert response == SimilarityResponse(responses=[], token=new_record.id)

    def test_get_nearest_neighbors_no_group_id(self):
        """
        Tests get_nearest_neighbors when the request has no group id and the matching record has
        no group id. Assert that a new record was added and its id is returned.
        """
        with Session() as session:
            embedding = grouping_lookup().encode_text("stacktrace")
            grouping_request = GroupingRequest(
                project_id=1, stacktrace="stacktrace", message="message_1"
            )
            grouping_lookup().insert_new_grouping_record(session, grouping_request, embedding)
            session.commit()

        grouping_request = GroupingRequest(
            project_id=1, stacktrace=b"stacktrace", message="message", k=1, threshold=0.01
        )

        response = grouping_lookup().get_nearest_neighbors(grouping_request)
        with Session() as session:
            new_record = session.query(DbGroupingRecord).filter_by(message="message").first()

        assert new_record
        assert response == SimilarityResponse(responses=[], token=new_record.id)

    def test_insert_new_grouping_record_group_record_exists(self):
        """
        Tests that insert_new_grouping_record only creates one record per group id.
        """
        with Session() as session:
            embedding = grouping_lookup().encode_text("stacktrace")
            grouping_request = GroupingRequest(
                group_id=1,
                project_id=1,
                stacktrace="stacktrace",
                message="message",
                stacktrace_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            # Insert the grouping record
            insert_id = grouping_lookup().insert_new_grouping_record(
                session, grouping_request, embedding
            )
            # Re-insert the grouping record
            existing_id = grouping_lookup().insert_new_grouping_record(
                session, grouping_request, embedding
            )
            assert insert_id == existing_id
