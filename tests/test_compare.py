from kllm.compare import _classic_forward_layer, _logic_forward_layer, print_report


class TestPrintReport:
    def test_does_not_raise(self, capsys):
        stats = {
            "text": "Hello",
            "tokens": 1,
            "layers": 2,
            "classic_time_s": 0.1234,
            "logic_time_s": 0.0567,
            "speedup": 2.18,
            "classic_peak_mb": 12.5,
            "logic_peak_mb": 3.2,
            "classic_output_shape": (1, 2048),
            "logic_output_shape": (1, 2048),
            "classic_output_sample": [0.1, 0.2, 0.3, 0.4, 0.5],
            "logic_output_sample": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
        print_report(stats)
        captured = capsys.readouterr()
        assert "Classic FP32 vs Lossless Bit-Sliced Logic" in captured.out
        assert "Hello" in captured.out
        assert "Speedup" in captured.out
