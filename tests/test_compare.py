from kllm.compare import print_report


class TestPrintReport:
    def test_does_not_raise(self, capsys):
        stats = {
            "text": "Hello",
            "tokens": 1,
            "layers": 2,
            "classic_time_s": 0.1234,
            "kllm_time_s": 0.0567,
            "speedup": 2.18,
            "classic_peak_mb": 12.5,
            "kllm_peak_mb": 3.2,
            "model_size_mb": 4200.0,
            "fabric_size_mb": 38.5,
            "classic_output_shape": (4, 32000),
            "kllm_output_shape": (4, 32000),
            "classic_decoded": "Hello world example",
            "kllm_decoded": "Hello world example",
            "classic_sample": [0.1, 0.2, 0.3, 0.4, 0.5],
            "kllm_sample": [0.1, 0.2, 0.3, 0.4, 0.5],
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "logits_match": True,
            "tokens_match": True,
        }
        print_report(stats)
        captured = capsys.readouterr()
        assert "HuggingFace vs Lossless Fabric" in captured.out
        assert "Hello" in captured.out
        assert "Logits match" in captured.out
