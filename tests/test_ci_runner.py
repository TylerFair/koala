"""Guard CI coverage and ensure infrastructure cannot hide real failures."""
import pytest

from run_ci import acceptable_exit, discover_files, junit_counts, make_shards


def test_shards_cover_new_and_known_files_once_and_are_deterministic():
    paths = ["a.py", "b.py", "new.py", "d.py"]
    weights = {"a.py": 90, "b.py": 60, "d.py": 30}
    groups, totals = make_shards(paths, weights, 3)
    assert sorted(p for group in groups for p in group) == sorted(paths)
    assert make_shards(list(reversed(paths)), weights, 3) == (groups, totals)
    assert max(totals) <= 90


@pytest.mark.parametrize("paths,count", [(["a.py", "a.py"], 2), (["a.py"], 0)])
def test_invalid_shard_plans_are_rejected(paths, count):
    with pytest.raises(ValueError):
        make_shards(paths, {}, count)


def test_collection_errors_and_failed_tests_cannot_be_accepted(tmp_path):
    xml = tmp_path / "result.xml"
    xml.write_text('<testsuites><testsuite><testcase><failure/></testcase>'
                   '<testcase><error/></testcase><testcase><skipped/></testcase>'
                   '<testcase/></testsuite></testsuites>')
    counts = junit_counts(xml)
    assert counts == {"passed": 1, "failed": 1, "errors": 1, "skipped": 1}
    assert not acceptable_exit(0, counts)
    assert not acceptable_exit(5, counts)


def test_only_expected_empty_or_skip_only_files_may_return_no_tests():
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 1}
    assert acceptable_exit(5, counts)
    assert not acceptable_exit(1, counts)
    assert not acceptable_exit(-9, counts)
    assert not acceptable_exit(5, {**counts, "passed": 1})


def test_discovery_includes_nested_new_tests_without_including_helpers(tmp_path):
    directory = tmp_path / "tests" / "nested"
    directory.mkdir(parents=True)
    (directory / "test_new.py").touch()
    (directory / "helper.py").touch()
    (tmp_path / "tests" / "test_root.py").touch()
    assert discover_files(tmp_path) == ["tests/nested/test_new.py", "tests/test_root.py"]
