import json
import threading
import urllib.error
import urllib.request
import pytest

from usual.autopilot import Store
from usual.review_server import make_server


@pytest.fixture
def review(tmp_path):
    store = Store(tmp_path / 'review.sqlite3')
    run = store.start('A local test run', 'scope')
    c = store.consult(run['id'], 'Which mascot name?', ['Pogo', 'Squiggle'])
    d = store.record(c['id'], 'Pogo', 'A reversible default.', [], 'low', 'agent_default')
    server, token = make_server(store, run['id'])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield store, f'http://127.0.0.1:{server.server_port}', token, d
    server.shutdown()
    server.server_close()


def test_data_requires_capability_and_correct_origin(review):
    store, base, token, d = review
    with urllib.request.urlopen(base) as response:
        assert token not in response.read().decode()
        assert response.headers['Cache-Control'] == 'no-store'
    for headers in [{}, {'Authorization':'Bearer wrong'}, {'Authorization':'Bearer '+token,'Origin':'https://attacker.invalid'}, {'Authorization':'Bearer '+token,'Host':'attacker.invalid'}]:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(base+'/api/report',headers=headers))
        assert error.value.code in (401,403)
    with urllib.request.urlopen(urllib.request.Request(base+'/api/report',headers={'Authorization':'Bearer '+token})) as response:
        assert json.load(response)['summary']['defaults'] == 1


def test_explicit_review_updates_only_current_run(review):
    store, base, token, d = review
    other_run = store.start('Another task','other')
    c = store.consult(other_run['id'],'Choose another mascot',['Pogo','Zippy'])
    other = store.record(c['id'],'Zippy','A default',[],'low','agent_default')
    headers = {'Authorization':'Bearer '+token,'Content-Type':'application/json'}
    payload = {'decision_id':other['id'],'verdict':'accepted'}
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(urllib.request.Request(base+'/api/review',data=json.dumps(payload).encode(),headers=headers))
    assert error.value.code == 400
    payload = {'decision_id':d['id'],'verdict':'corrected','statement':'Use Squiggle for the mascot name.'}
    with urllib.request.urlopen(urllib.request.Request(base+'/api/review',data=json.dumps(payload).encode(),headers=headers)) as response:
        assert json.load(response)['review'] == 'corrected'
    assert store.evidence('scope')[0]['call'] == payload['statement']


def test_review_rejects_missing_capability_and_invalid_body(review):
    _, base, token, d = review
    for headers, data in [({},b'{}'),({'Authorization':'Bearer '+token,'Content-Type':'text/plain'},b'{}'),({'Authorization':'Bearer '+token,'Content-Type':'application/json'},b'[]')]:
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(urllib.request.Request(base+'/api/review',data=data,headers=headers))
