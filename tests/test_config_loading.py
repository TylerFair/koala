"""Malformed YAML documents fail at the public configuration boundary."""
import pytest
from koala.config import load_config


@pytest.mark.parametrize('text', ['', '# no sections\n', '- planet\n', '42\n', 'null\n'])
def test_non_mapping_config_has_actionable_error(tmp_path, text):
    path = tmp_path / 'observation.yaml'
    path.write_text(text)
    with pytest.raises(ValueError, match='expected a YAML mapping.*examples/niriss_soss_order1.yaml') as error:
        load_config(path)
    assert str(path) in str(error.value)


def test_mapping_config_preserves_yaml_values(tmp_path):
    path = tmp_path / 'observation.yaml'
    path.write_text('flags: {random_seed: 7}\nplanet: {name: example}\n')
    assert load_config(path) == {'flags': {'random_seed': 7}, 'planet': {'name': 'example'}}
