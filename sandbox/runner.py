"""Container-only runner: stdin is the only input, stdout is one JSON response."""
import json
import sys

payload = json.loads(sys.stdin.read())
code, grid = payload['code'], payload['grid']
namespace = {}
exec(compile(code, '<candidate>', 'exec'), namespace)
result = namespace['transform'](grid)
sys.stdout.write(json.dumps({'grid': result}, separators=(',', ':')))
