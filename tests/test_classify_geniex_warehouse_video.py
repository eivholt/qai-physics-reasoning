import io
import json
import unittest
from unittest import mock

from scripts import classify_geniex_warehouse_video as classifier


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


class SequenceUrlOpen:
    def __init__(self, letters: list[str]):
        self.letters = iter(letters)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        letter = next(self.letters)
        body = {"choices": [{"message": {"content": letter}}]}
        return _Response(json.dumps(body).encode("utf-8"))


class ClassifyGenieXWarehouseVideoTests(unittest.TestCase):
    def test_all_decision_tree_leaves(self):
        cases = {
            ("A", "A"): "forklift_marker_knockdown",
            ("A", "B"): "forklift_human_near_miss",
            ("B", "A"): "routine_box_pickup",
            ("B", "B"): "multi_worker_aisle_exit",
        }
        for answers, expected in cases.items():
            with self.subTest(answers=answers):
                opener = SequenceUrlOpen(list(answers))
                with mock.patch.object(
                    classifier.time, "monotonic", side_effect=[1.0, 2.0, 3.0, 5.0]
                ):
                    result = classifier.classify_video(
                        server_url="http://127.0.0.1:18181/",
                        video_url="file:///tmp/clip.mp4",
                        timeout_seconds=12.0,
                        urlopen=opener,
                    )

                self.assertEqual(result["event"], expected)
                self.assertEqual(result["request_seconds"], [1.0, 2.0])
                self.assertEqual(result["total_seconds"], 3.0)
                self.assertEqual(len(opener.requests), 2)

                first_request, timeout = opener.requests[0]
                self.assertEqual(
                    first_request.full_url,
                    "http://127.0.0.1:18181/v1/chat/completions",
                )
                self.assertEqual(timeout, 12.0)
                self.assertEqual(first_request.get_header("Connection"), "close")
                first_payload = json.loads(first_request.data)
                self.assertEqual(
                    first_payload["model"], "local/cosmos-reason2-2b:Q4_0"
                )
                self.assertEqual(first_payload["grammar_string"], "root ::= [AB]")
                self.assertEqual(
                    first_payload["messages"][0]["content"][0]["image_url"]["url"],
                    "file:///tmp/clip.mp4",
                )

                second_payload = json.loads(opener.requests[1][0].data)
                expected_prompt = (
                    classifier.FORKLIFT_PROMPT
                    if answers[0] == "A"
                    else classifier.WORKER_PROMPT
                )
                self.assertEqual(
                    second_payload["messages"][0]["content"][1]["text"],
                    expected_prompt,
                )

    def test_rejects_out_of_grammar_response(self):
        opener = SequenceUrlOpen(["C"])
        with self.assertRaisesRegex(ValueError, "out-of-grammar"):
            classifier.classify_video(
                server_url="http://127.0.0.1:18181",
                video_url="file:///tmp/clip.mp4",
                urlopen=opener,
            )

    def test_parser_requires_one_video_source(self):
        parser = classifier.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--video", "a.mp4", "--video-url", "file:///b.mp4"])


if __name__ == "__main__":
    unittest.main()
