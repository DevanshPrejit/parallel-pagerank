import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'convert_snap', Path(__file__).resolve().parents[1] / 'scripts/convert_snap.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class ConverterTests(unittest.TestCase):
    def test_conversion_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / 'raw.txt'
            output = Path(folder) / 'graph.txt'
            raw.write_text('# comment\n100 900\n100 900\n900 900 # loop\n')
            with contextlib.redirect_stdout(io.StringIO()):
                converter.convert(raw, output)
            self.assertEqual(output.read_text(), '2 3\n0 1\n0 1\n1 1\n')
            with self.assertRaises(FileExistsError):
                converter.convert(raw, output)
            self.assertEqual(output.read_text(), '2 3\n0 1\n0 1\n1 1\n')

    def test_invalid_input_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / 'raw.txt'
            output = Path(folder) / 'graph.txt'
            for value in ['1 x\n', '-1 2\n', '1 2 3\n', '# empty\n']:
                raw.write_text(value)
                with self.assertRaises(ValueError):
                    converter.convert(raw, output)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
