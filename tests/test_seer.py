import datetime
import json
import os
import unittest
from unittest import mock

import pytest
from johen.pytest import parametrize
from sqlalchemy import text

from seer.app import app
from seer.db import AsyncSession, DbGroupingRecord, ProcessRequest, Session


@pytest.fixture(autouse=True)
def mock_severity_score():
    # Create a mock instance with a dummy severity_score method
    with mock.patch("seer.inference_models.SeverityInference") as mock_severity_inference:
        mock_instance = mock_severity_inference.return_value
        mock_instance.severity_score.return_value = [0, 1]
        yield


class TestSeer(unittest.TestCase):
    def get_sample_data(self):
        trend_data = {
            "data": [
                [1681934400, [{"count": 680.0}]],
                [1681938000, [{"count": 742.0}]],
                [1681941600, [{"count": 631.0}]],
                [1681945200, [{"count": 567.0}]],
                [1681948800, [{"count": 538.0}]],
                [1681952400, [{"count": 619.0}]],
                [1681956000, [{"count": 577.0}]],
                [1681959600, [{"count": 551.5}]],
                [1681963200, [{"count": 589.0}]],
                [1681966800, [{"count": 531.5}]],
                [1681970400, [{"count": 562.0}]],
                [1681974000, [{"count": 528.0}]],
                [1681977600, [{"count": 587.0}]],
                [1681981200, [{"count": 569.0}]],
                [1681984800, [{"count": 615.0}]],
                [1681988400, [{"count": 611.0}]],
                [1681992000, [{"count": 630.0}]],
                [1681995600, [{"count": 637.0}]],
                [1681999200, [{"count": 661.0}]],
                [1682002800, [{"count": 671.0}]],
                [1682006400, [{"count": 638.0}]],
                [1682010000, [{"count": 670.0}]],
                [1682013600, [{"count": 666.5}]],
                [1682017200, [{"count": 650.0}]],
                [1682020800, [{"count": 641.0}]],
                [1682024400, [{"count": 682.0}]],
                [1682028000, [{"count": 627.0}]],
                [1682031600, [{"count": 611.5}]],
                [1682035200, [{"count": 557.0}]],
                [1682038800, [{"count": 526.0}]],
                [1682042400, [{"count": 538.0}]],
                [1682046000, [{"count": 598.0}]],
                [1682049600, [{"count": 563.0}]],
                [1682053200, [{"count": 588.0}]],
                [1682056800, [{"count": 610.5}]],
                [1682060400, [{"count": 576.0}]],
                [1682064000, [{"count": 598.0}]],
                [1682067600, [{"count": 560.5}]],
                [1682071200, [{"count": 623.0}]],
                [1682074800, [{"count": 557.0}]],
                [1682078400, [{"count": 883.5}]],
                [1682082000, [{"count": 972.0}]],
                [1682085600, [{"count": 844.5}]],
                [1682089200, [{"count": 929.0}]],
                [1682092800, [{"count": 1071.0}]],
                [1682096400, [{"count": 1090.0}]],
                [1682100000, [{"count": 883.0}]],
                [1682103600, [{"count": 913.0}]],
                [1682107200, [{"count": 850.0}]],
                [1682110800, [{"count": 911.5}]],
                [1682114400, [{"count": 814.0}]],
                [1682118000, [{"count": 786.0}]],
                [1682121600, [{"count": 660.5}]],
                [1682125200, [{"count": 605.5}]],
                [1682128800, [{"count": 551.0}]],
                [1682132400, [{"count": 430.0}]],
                [1682136000, [{"count": 635.0}]],
                [1682139600, [{"count": 569.0}]],
                [1682143200, [{"count": 491.5}]],
                [1682146800, [{"count": 536.0}]],
                [1682150400, [{"count": 533.0}]],
                [1682154000, [{"count": 393.0}]],
                [1682157600, [{"count": 534.0}]],
                [1682161200, [{"count": 498.0}]],
                [1682164800, [{"count": 645.5}]],
                [1682168400, [{"count": 521.0}]],
                [1682172000, [{"count": 485.5}]],
                [1682175600, [{"count": 668.0}]],
                [1682179200, [{"count": 654.0}]],
                [1682182800, [{"count": 520.5}]],
                [1682186400, [{"count": 619.5}]],
                [1682190000, [{"count": 549.5}]],
                [1682193600, [{"count": 560.0}]],
                [1682197200, [{"count": 550.5}]],
                [1682200800, [{"count": 604.5}]],
                [1682204400, [{"count": 623.0}]],
                [1682208000, [{"count": 561.0}]],
                [1682211600, [{"count": 598.0}]],
                [1682215200, [{"count": 743.5}]],
                [1682218800, [{"count": 658.0}]],
                [1682222400, [{"count": 704.0}]],
                [1682226000, [{"count": 606.0}]],
                [1682229600, [{"count": 508.0}]],
                [1682233200, [{"count": 486.0}]],
                [1682236800, [{"count": 554.0}]],
                [1682240400, [{"count": 543.0}]],
                [1682244000, [{"count": 435.0}]],
                [1682247600, [{"count": 561.5}]],
                [1682251200, [{"count": 518.0}]],
                [1682254800, [{"count": 661.0}]],
                [1682258400, [{"count": 514.5}]],
                [1682262000, [{"count": 581.5}]],
                [1682265600, [{"count": 503.0}]],
                [1682269200, [{"count": 598.0}]],
                [1682272800, [{"count": 520.5}]],
                [1682276400, [{"count": 494.0}]],
                [1682280000, [{"count": 785.0}]],
                [1682283600, [{"count": 383.0}]],
                [1682287200, [{"count": 457.0}]],
                [1682290800, [{"count": 464.0}]],
                [1682294400, [{"count": 559.0}]],
                [1682298000, [{"count": 489.5}]],
                [1682301600, [{"count": 746.0}]],
                [1682305200, [{"count": 609.0}]],
                [1682308800, [{"count": 587.0}]],
                [1682312400, [{"count": 1263.5}]],
                [1682316000, [{"count": 744.5}]],
                [1682319600, [{"count": 805.5}]],
                [1682323200, [{"count": 987.0}]],
                [1682326800, [{"count": 869.0}]],
                [1682330400, [{"count": 779.5}]],
                [1682334000, [{"count": 880.5}]],
                [1682337600, [{"count": 929.5}]],
                [1682341200, [{"count": 862.0}]],
                [1682344800, [{"count": 884.0}]],
                [1682348400, [{"count": 895.0}]],
                [1682352000, [{"count": 939.0}]],
                [1682355600, [{"count": 1183.0}]],
                [1682359200, [{"count": 922.0}]],
                [1682362800, [{"count": 953.0}]],
                [1682366400, [{"count": 1373.5}]],
                [1682370000, [{"count": 963.0}]],
                [1682373600, [{"count": 719.5}]],
                [1682377200, [{"count": 1024.5}]],
                [1682380800, [{"count": 940.0}]],
                [1682384400, [{"count": 630.0}]],
                [1682388000, [{"count": 943.0}]],
                [1682391600, [{"count": 796.5}]],
                [1682395200, [{"count": 695.5}]],
                [1682398800, [{"count": 965.5}]],
                [1682402400, [{"count": 921.5}]],
                [1682406000, [{"count": 896.0}]],
                [1682409600, [{"count": 962.0}]],
                [1682413200, [{"count": 1099.0}]],
                [1682416800, [{"count": 837.0}]],
                [1682420400, [{"count": 915.0}]],
                [1682424000, [{"count": 978.5}]],
                [1682427600, [{"count": 1051.5}]],
                [1682431200, [{"count": 1125.0}]],
                [1682434800, [{"count": 838.5}]],
                [1682438400, [{"count": 936.0}]],
                [1682442000, [{"count": 1170.0}]],
                [1682445600, [{"count": 1057.5}]],
                [1682449200, [{"count": 1097.0}]],
                [1682452800, [{"count": 1034.0}]],
                [1682456400, [{"count": 1219.0}]],
                [1682460000, [{"count": 936.0}]],
                [1682463600, [{"count": 911.0}]],
                [1682467200, [{"count": 841.0}]],
                [1682470800, [{"count": 790.0}]],
                [1682474400, [{"count": 1015.0}]],
                [1682478000, [{"count": 651.5}]],
                [1682481600, [{"count": 839.0}]],
                [1682485200, [{"count": 820.0}]],
                [1682488800, [{"count": 783.0}]],
                [1682492400, [{"count": 853.0}]],
                [1682496000, [{"count": 811.0}]],
                [1682499600, [{"count": 971.0}]],
                [1682503200, [{"count": 931.0}]],
                [1682506800, [{"count": 1028.0}]],
                [1682510400, [{"count": 828.0}]],
                [1682514000, [{"count": 817.0}]],
                [1682517600, [{"count": 971.0}]],
                [1682521200, [{"count": 1235.0}]],
                [1682524800, [{"count": 1080.0}]],
                [1682528400, [{"count": 974.0}]],
                [1682532000, [{"count": 1016.0}]],
                [1682535600, [{"count": 938.0}]],
                [1682539200, [{"count": 738.5}]],
                [1682542800, [{"count": 924.0}]],
                [1682546400, [{"count": 900.0}]],
                [1682550000, [{"count": 958.0}]],
                [1682553600, [{"count": 974.0}]],
                [1682557200, [{"count": 756.0}]],
                [1682560800, [{"count": 912.0}]],
                [1682564400, [{"count": 924.0}]],
                [1682568000, [{"count": 822.0}]],
                [1682571600, [{"count": 776.0}]],
                [1682575200, [{"count": 979.0}]],
                [1682578800, [{"count": 606.0}]],
                [1682582400, [{"count": 1109.5}]],
                [1682586000, [{"count": 884.5}]],
                [1682589600, [{"count": 833.0}]],
                [1682593200, [{"count": 897.0}]],
                [1682596800, [{"count": 844.0}]],
                [1682600400, [{"count": 1014.0}]],
            ],
            "request_start": 1681934400,
            "request_end": 1683144000,
            "data_start": 1681934400,
            "data_end": 1683144000,
        }
        input_data = {
            "data": {"sentry,/api/0/organizations/{organization_slug}/issues/": trend_data},
            "sort": "-trend_percentage()",
        }
        return input_data

    def test_empty_txns_dataset(self):
        response = app.test_client().post(
            "/trends/breakpoint-detector",
            data=json.dumps({"data": {}, "sort": "trend_percentage()"}),
            content_type="application/json",
        )

        actual_output = json.loads(response.get_data(as_text=True))

        expected_output = {"data": []}

        assert actual_output == expected_output

    def test_breakpoint_output(self):
        input_data = self.get_sample_data()

        response = app.test_client().post(
            "/trends/breakpoint-detector",
            data=json.dumps(input_data),
            content_type="application/json",
        )

        assert response.status_code == 200

        request_start = input_data["data"][
            "sentry,/api/0/organizations/{organization_slug}/issues/"
        ]["request_start"]
        request_end = input_data["data"]["sentry,/api/0/organizations/{organization_slug}/issues/"][
            "request_end"
        ]

        expected_output = {
            "data": [
                {
                    "project": "sentry",
                    "transaction": "/api/0/organizations/{organization_slug}/issues/",
                    "data_start": 1681934400,
                    "data_end": 1683144000,
                    "aggregate_range_1": 619.2740384615385,
                    "aggregate_range_2": 921.5060975609756,
                    "unweighted_t_value": -14.710852035668289,
                    "unweighted_p_value": round(7.386632642605361e-32, 10),
                    "trend_percentage": 1.4880425148295766,
                    "trend_difference": 302.23205909943715,
                    "breakpoint": 1682308800,
                    "change": "regression",
                    "absolute_percentage_change": 1.4880425148295766,
                    "request_start": request_start,
                    "request_end": request_end,
                }
            ]
        }

        actual_output = json.loads(response.get_data(as_text=True))

        assert actual_output == expected_output

    def test_no_data_after_request_start(self):
        mid = 5  # needs to be greater than 3 because that's the minimum time series length
        input_data = {
            "data": {
                "sentry,/api/0/organizations/{organization_slug}/issues/": {
                    "data": [[ts, [{"count": 1 if ts <= mid else 0}]] for ts in range(2 * mid)],
                    "request_start": mid,
                    "request_end": 2 * mid,
                    "data_start": 0,
                    "data_end": 2 * mid,
                },
            },
            "sort": "-trend_percentage()",
        }

        response = app.test_client().post(
            "/trends/breakpoint-detector",
            data=json.dumps(input_data),
            content_type="application/json",
        )

        output = json.loads(response.get_data(as_text=True))
        assert output == {"data": []}

    def test_similarity_grouping_record_endpoint_valid(self):
        """Test the similarity grouping record endpoint"""
        record_requests = {
            "data": [
                {
                    "group_id": i,
                    "project_id": 1,
                    "message": "message " + str(i),
                }
                for i in range(5)
            ],
            "stacktrace_list": ["stacktrace " + str(i) for i in range(5)],
        }

        response = app.test_client().post(
            "/v0/issues/similar-issues/grouping-record",
            data=json.dumps(record_requests),
            content_type="application/json",
        )
        output = json.loads(response.get_data(as_text=True))
        assert output == {"success": True}
        for i in range(5):
            with Session() as session:
                assert (
                    session.query(DbGroupingRecord).filter(DbGroupingRecord.group_id == i).first()
                    is not None
                )

    def test_similarity_grouping_record_endpoint_valid(self):
        """
        Test the similarity grouping record endpoint is unsuccessful when input lists are of
        different lengths
        """
        record_requests = {
            "data": [
                {
                    "group_id": i,
                    "project_id": 1,
                    "message": "message " + str(i),
                }
                for i in range(2)
            ],
            "stacktrace_list": ["stacktrace " + str(i) for i in range(3)],
        }

        response = app.test_client().post(
            "/v0/issues/similar-issues/grouping-record",
            data=json.dumps(record_requests),
            content_type="application/json",
        )
        output = json.loads(response.get_data(as_text=True))
        assert output == {"success": False}
        for i in range(2):
            with Session() as session:
                assert (
                    session.query(DbGroupingRecord).filter(DbGroupingRecord.group_id == i).first()
                    is None
                )


@parametrize(count=1)
def test_prepared_statements_disabled(
    requests: tuple[
        ProcessRequest,
        ProcessRequest,
        ProcessRequest,
        ProcessRequest,
        ProcessRequest,
        ProcessRequest,
    ]
):
    with Session() as session:
        # This would cause postgresql to issue prepared statements.  Remove logic from bootup connect args to validate.
        for i, request in enumerate(requests):
            request.name += str(i)
            session.add(request)
            session.flush()
        assert session.execute(text("select count(*) from pg_prepared_statements")).scalar() == 0


@pytest.mark.asyncio
@parametrize(count=1)
async def test_async_prepared_statements_disabled(
    requests: tuple[
        ProcessRequest,
        ProcessRequest,
        ProcessRequest,
        ProcessRequest,
        ProcessRequest,
        ProcessRequest,
    ]
):
    async with AsyncSession() as session:
        # This would cause postgresql to issue prepared statements.  Remove logic from bootup connect args to validate.
        for i, request in enumerate(requests):
            request.name += str(i)
            session.add(request)
            await session.flush()
        assert (
            await session.execute(text("select count(*) from pg_prepared_statements"))
        ).scalar() == 0
