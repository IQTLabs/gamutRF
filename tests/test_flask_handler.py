#!/usr/bin/python3
import unittest

from gamutrf.flask_handler import FlaskHandler


class FakeOptions:
    def __init__(self, apiport, foo, bar):
        self.apiport = apiport
        self.foo = foo
        self.bar = bar


class FakeRequest:
    def __init__(self, args):
        self.args = args


class FlaskHandlerTestCase(unittest.TestCase):
    def test_flask_handler(self):
        def good_check_options(new_options):
            return ""

        def bad_check_options(new_options):
            return "bad options are bad"

        banned_args = []
        options = FakeOptions(apiport=2048, foo=123, bar=None)
        request = FakeRequest(args={"foo": "999", "bar": None, "apiport": 123})
        handler = FlaskHandler(options, bad_check_options, banned_args)
        handler.request = request
        self.assertEqual(("bad options are bad", 400), handler.reconf())
        self.assertEqual(handler.options, options)
        handler.check_options = good_check_options
        self.assertEqual(("reconf", 200), handler.reconf())
        self.assertNotEqual(handler.options, options)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
