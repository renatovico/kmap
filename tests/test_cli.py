from kllm.cli import _build_parser


class TestCLIParser:
    def test_compile_mode(self):
        args = _build_parser().parse_args(["--mode", "compile"])
        assert args.mode == "compile"
        assert args.model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert args.save_dir == "./lossless_logic"
        assert args.solver_timeout == 200

    def test_inference_mode_with_text(self):
        args = _build_parser().parse_args(["--mode", "inference", "--text", "Hello"])
        assert args.mode == "inference"
        assert args.text == "Hello"

    def test_compare_mode(self):
        args = _build_parser().parse_args(["--mode", "compare", "--text", "Hi"])
        assert args.mode == "compare"
        assert args.text == "Hi"

    def test_full_mode(self):
        args = _build_parser().parse_args(["--mode", "full"])
        assert args.mode == "full"

    def test_custom_solver_timeout(self):
        args = _build_parser().parse_args(["--mode", "compile", "--solver-timeout", "500"])
        assert args.solver_timeout == 500

    def test_custom_max_layers(self):
        args = _build_parser().parse_args(["--mode", "compare", "--max-layers", "3", "--text", "x"])
        assert args.max_layers == 3

    def test_max_layers_default_none(self):
        args = _build_parser().parse_args(["--mode", "compile"])
        assert args.max_layers is None

    def test_custom_model(self):
        args = _build_parser().parse_args(["--mode", "compile", "--model", "my/model"])
        assert args.model == "my/model"

    def test_custom_save_dir(self):
        args = _build_parser().parse_args(["--mode", "compile", "--save-dir", "/tmp/out"])
        assert args.save_dir == "/tmp/out"
