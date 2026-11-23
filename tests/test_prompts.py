from app.inference import OpenWorldEngine


def test_parse_prompts_preserves_natural_language_commas():
    prompts = OpenWorldEngine.parse_prompts(
        "person wearing a red helmet\nsmall drone, flying low;yellow safety cone"
    )
    assert prompts == [
        "person wearing a red helmet",
        "small drone, flying low",
        "yellow safety cone",
    ]


def test_parse_prompts_deduplicates_case_insensitively():
    prompts = OpenWorldEngine.parse_prompts("Drone\ndrone\nred car")
    assert prompts == ["Drone", "red car"]
