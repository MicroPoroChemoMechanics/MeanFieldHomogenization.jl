# The text conventions of the scripts: floats as Python's `repr` writes them,
# numbers as Julia code, and the embedded model as Python's `json.dumps(…,
# indent = 1, sort_keys = True)`. The expected strings are Python's own output
# (Python 3.12), so that the scripts the Python studio saved read back the same.

reprs = [
    0.1 => "0.1", 1.0 => "1.0", -0.0 => "-0.0", 0.0 => "0.0", 1.0e-6 => "1e-06",
    1.0e-5 => "1e-05", 0.0001 => "0.0001", 0.0001234 => "0.0001234",
    1.0e15 => "1000000000000000.0", 1.0e16 => "1e+16", 123456789.0 => "123456789.0",
    1.0e22 => "1e+22", 2.5e-12 => "2.5e-12", 1 / 3 => "0.3333333333333333",
    1.0e300 => "1e+300", -1.5e-7 => "-1.5e-07", 5.0e-324 => "5e-324",
    floatmax(Float64) => "1.7976931348623157e+308", 100.0 => "100.0",
    12345678901234567.0 => "1.2345678901234568e+16", 0.5 => "0.5", 12.25 => "12.25",
    -3.5 => "-3.5", 1234.5678 => "1234.5678",
    Inf => "inf", -Inf => "-inf", NaN => "nan",
]
for (x, s) in reprs
    @test MS.py_repr(x) == s
end
# every float reads back as itself
for x in (0.1, 1 / 3, 1.0e-5, 2.5e-12, 1.0e16, 123456789.0, 5.0e-324)
    @test parse(Float64, MS.py_repr(x)) === x
end

# as Julia code: float-ness explicit, the exponent without its padding
@test MS.julia_number(1.0e-6) == "1.0e-6"
@test MS.julia_number(1.0e16) == "1.0e16"
@test MS.julia_number(-1.5e-7) == "-1.5e-7"
@test MS.julia_number(2.5e-12) == "2.5e-12"
@test MS.julia_number(0.5) == "0.5"
@test MS.julia_number(3) == "3"
@test MS.julia_number(true) == "true" && MS.julia_number(false) == "false"
@test MS.julia_number("π / 4") == "π / 4"
@test MS.julia_number(nothing) == "None"
@test MS.julia_number(Inf) == "inf.0"           # as the Python studio wrote it
@test MS.julia_float(1) == "1.0" && MS.julia_float(true) == "true" && MS.julia_float("a") == "a"
@test MS.julia_float(0.25) == "0.25"

# Python's truth values and `or`
@test !MS.truthy(nothing) && !MS.truthy(false) && !MS.truthy(0) && !MS.truthy(0.0)
@test !MS.truthy("") && !MS.truthy(Any[]) && !MS.truthy(MS.JDict())
@test MS.truthy("x") && MS.truthy(1) && MS.truthy(Any[0]) && MS.truthy(:x)
@test MS.either("", "d") == "d" && MS.either("x", "d") == "x" && MS.either(nothing, 1) == 1

# json.dumps(indent = 1, sort_keys = True), ensure_ascii
value = MS.jdict(
    "b" => Any[1, 1.0, true, nothing, "φ\"\\\n\x7f😀"], "a" => MS.JDict(), "c" => Any[],
    "d" => MS.jdict("z" => 1.0e-6, "y" => Inf),
)
expected = """
{
 "a": {},
 "b": [
  1,
  1.0,
  true,
  null,
  "\\u03c6\\"\\\\\\n\\u007f\\ud83d\\ude00"
 ],
 "c": [],
 "d": {
  "y": Infinity,
  "z": 1e-06
 }
}"""
@test MS.py_json(value) == expected
@test MS.py_json(Any[-Inf, NaN, "\r\t\b\f"]) == "[\n -Infinity,\n NaN,\n \"\\r\\t\\b\\f\"\n]"
@test MS.py_json((1, 2)) == "[\n 1,\n 2\n]"

# f-string interpolation of a value of the model
@test MS.py_str(nothing) == "None" && MS.py_str(true) == "True" && MS.py_str(1.0e-6) == "1e-06"
@test MS.py_str(Any[1, "a", nothing]) == "[1, 'a', None]"
@test MS.py_str(MS.jdict("k" => 1)) == "{'k': 1}"
