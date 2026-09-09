from examples.benchmark_sms import prepare


def test_benchmark_deduplicates_and_groups_digit_variants(tmp_path):
    source = tmp_path / "sms"
    source.write_text("ham\tMeet at 12\nham\t MEET  at 12 \nham\tMeet at 34\nspam\tFree prize\n")
    rows, counts = prepare(source)
    assert counts["source_rows"] == 4
    assert counts["normalized_unique"] == 3
    ham = [row for row in rows if row["label"] == "ham"]
    assert ham[0]["group"] == ham[1]["group"]


def test_benchmark_excludes_conflicting_labels(tmp_path):
    source = tmp_path / "sms"
    source.write_text("ham\tCall me\nspam\tCALL ME\nham\tSee you\n")
    rows, counts = prepare(source)
    assert counts["conflicting_texts_excluded"] == 1
    assert len(rows) == 1
    assert rows[0]["text"] == "See you"
