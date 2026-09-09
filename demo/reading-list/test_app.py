import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from app import make_server


class ReadingListTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name)/'reading.sqlite3'
        self.start()

    def start(self):
        self.server, self.token = make_server(self.db)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def tearDown(self):
        self.stop()
        self.temp.cleanup()

    def request(self, path='/api/items', data=None, headers=None):
        hdr = {'Authorization': 'Bearer '+self.token, 'Content-Type':'application/json'}
        hdr.update(headers or {})
        req = Request(self.base+path, data=json.dumps(data).encode() if data is not None else None, headers=hdr)
        with urlopen(req, timeout=5) as response:
            return json.load(response)

    def test_add_mark_and_restart_persistence(self):
        self.request(data={'title':'A useful article', 'url':'https://example.com/article'})
        item = self.request()[0]
        self.assertEqual(item['title'], 'A useful article')
        self.assertEqual(item['is_read'], 0)
        self.request('/api/update', {'id':item['id'], 'is_read':True})
        self.stop()
        self.start()
        self.assertEqual(self.request()[0]['is_read'], 1)
        self.request('/api/update', {'id':item['id'], 'is_read':False})
        self.assertEqual(self.request()[0]['is_read'], 0)

    def test_private_data_and_mutations_require_capability_and_origin(self):
        for headers, code in [({'Authorization':''},401), ({'Origin':'https://evil.example'},403), ({'Host':'evil.example'},403)]:
            with self.assertRaises(HTTPError) as error:
                self.request(headers=headers)
            self.assertEqual(error.exception.code, code)

    def test_invalid_links_and_updates_do_not_write(self):
        for data in [{'title':'','url':'https://example.com'}, {'title':'Bad','url':'javascript:alert(1)'}, {'title':'Bad','url':'https://user:pass@example.com'}]:
            with self.assertRaises(HTTPError) as error:
                self.request(data=data)
            self.assertEqual(error.exception.code, 400)
        self.assertEqual(self.request(), [])
        with self.assertRaises(HTTPError) as error:
            self.request('/api/update', {'id':999,'is_read':True})
        self.assertEqual(error.exception.code, 404)


if __name__ == '__main__':
    unittest.main()
