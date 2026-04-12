from kllm.compare import print_generate_report


class TestPrintReport:
    def test_does_not_raise(self, capsys):
        stats = {
            "text": "Hello",
            "max_tokens": 3,
            "hf_time_s": 0.5,
            "kllm_time_s": 1.2,
            "hf_output": "I am a",
            "kllm_output": "I am a",
        }
        print_generate_report(stats)
        captured = capsys.readouterr()
        assert "HuggingFace" in captured.out
        assert "Hello" in captured.out
