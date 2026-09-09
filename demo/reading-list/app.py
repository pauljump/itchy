#!/usr/bin/env python3
"""A local reading list, built using the accompanying Usual decision receipts."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import sqlite3
from urllib.parse import urlsplit

PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reading room</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f1efe7;color:#25362c;font:17px/1.6 system-ui,sans-serif}main{max-width:760px;margin:64px auto;padding:0 24px}h1{font:52px/1.1 Georgia,serif;margin:10px 0}p{color:#58675d}label{display:block;margin-top:16px;font-weight:600}input,select,button{font:inherit;padding:12px;border:1px solid #a6b3a5;border-radius:6px}input{width:100%;background:#fffef8}button{cursor:pointer;background:#284d39;color:white;min-height:48px}button:disabled{opacity:.5}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #b26a21;outline-offset:3px}form{background:#e5e8da;padding:24px;border-radius:14px;margin:28px 0}form button{margin-top:20px}article{display:flex;gap:20px;justify-content:space-between;align-items:center;padding:22px 0;border-bottom:1px solid #b9c4b5}article div{min-width:0}article a{color:#274b35;font-weight:650;overflow-wrap:anywhere}article p{font-size:14px;margin:5px 0}article button{flex-shrink:0;background:transparent;color:#284d39}.top{display:flex;align-items:center;justify-content:space-between;gap:20px}.small{font-size:13px}#status{min-height:28px;color:#803a1a}@media(max-width:500px){main{margin-top:35px}h1{font-size:42px}article{align-items:start;flex-direction:column}.top{flex-wrap:wrap}}</style>
<main><span class="small">BUILT WITH USUAL</span><h1>Reading room.</h1><p>A quiet place for things you want to read. Saved on this computer.</p>
<form id="add"><label for="title">Title</label><input id="title" required maxlength="300" placeholder="Something worth coming back to"><label for="url">Article URL</label><input id="url" type="url" required maxlength="2048" placeholder="https://example.com/article"><button id="submit">Add to list</button></form>
<div class="top"><h2>Your reading list</h2><div><label for="filter" class="small">Show</label><select id="filter"><option value="all">All articles</option><option value="unread">Unread</option><option value="read">Read</option></select></div></div><p id="status" role="status"></p><div id="items" aria-live="polite"></div><p class="small">Local SQLite storage · No account · No tracking</p></main>
<script>'use strict';let token=location.hash.slice(1);history.replaceState(null,'',location.pathname);let items=[];const status=document.getElementById('status');
async function api(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers:{'Authorization':'Bearer '+token,'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});const data=await r.json();if(!r.ok)throw Error(data.error||'Request failed');return data;}
function render(){const list=document.getElementById('items');list.replaceChildren();const filter=document.getElementById('filter').value;const visible=items.filter(i=>filter==='all'||(filter==='read')===Boolean(i.is_read));if(!visible.length){const p=document.createElement('p');p.textContent=items.length?'No articles match this filter.':'Your next good read belongs here. Add an article above.';list.append(p);}for(const item of visible){const row=document.createElement('article'),text=document.createElement('div'),link=document.createElement('a'),meta=document.createElement('p'),button=document.createElement('button');link.textContent=item.title;link.href=item.url;link.target='_blank';link.rel='noopener noreferrer';meta.textContent=item.is_read?'Read':'Unread';button.textContent=item.is_read?'Mark unread':'Mark read';button.onclick=async()=>{button.disabled=true;try{await api('/api/update',{id:item.id,is_read:!item.is_read});await refresh();}catch(e){status.textContent=e.message;button.disabled=false;}};text.append(link,meta);row.append(text,button);list.append(row);}}
async function refresh(){items=await api('/api/items');render();}document.getElementById('filter').onchange=render;document.getElementById('add').onsubmit=async e=>{e.preventDefault();const button=document.getElementById('submit');button.disabled=true;status.textContent='';try{await api('/api/items',{title:document.getElementById('title').value,url:document.getElementById('url').value});e.target.reset();await refresh();status.textContent='Added to your list.';document.getElementById('title').focus();}catch(err){status.textContent=err.message;}finally{button.disabled=false;}};refresh().catch(e=>{status.textContent=token?e.message:'Open the full private link printed by the server to access your list.';});</script></html>'''


def connect(path):
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    return db


def make_server(path, port=0):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.touch(mode=0o600, exist_ok=False)
    except FileExistsError:
        pass
    with connect(path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS items(id INTEGER PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL, is_read INTEGER NOT NULL DEFAULT 0)')
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, data, content_type='application/json'):
            body = data.encode() if isinstance(data, str) else json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type + '; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(body)

        def permitted(self, private=True):
            hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            if self.headers.get('Host') not in hosts:
                self.send(403, {'error': 'Invalid host'})
                return False
            if self.headers.get('Origin') and self.headers['Origin'] not in {'http://' + h for h in hosts}:
                self.send(403, {'error': 'Invalid origin'})
                return False
            if private and not secrets.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token):
                self.send(401, {'error': 'Use the private link printed by the server.'})
                return False
            return True

        def do_GET(self):
            if not self.permitted(private=self.path != '/'):
                return
            if self.path == '/':
                self.send(200, PAGE, 'text/html')
            elif self.path == '/api/items':
                with connect(path) as db:
                    rows = [dict(row) for row in db.execute('SELECT * FROM items ORDER BY id DESC')]
                self.send(200, rows)
            else:
                self.send(404, {'error': 'Not found'})

        def do_POST(self):
            if not self.permitted():
                return
            try:
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('Send JSON')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 8192:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError('Send a JSON object')
                with connect(path) as db:
                    if self.path == '/api/items':
                        title, url = data.get('title'), data.get('url')
                        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 300:
                            raise ValueError('Enter a title up to 300 characters')
                        if not isinstance(url, str) or len(url) > 2048:
                            raise ValueError('Enter a valid URL')
                        parsed = urlsplit(url.strip())
                        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or any(ord(c) < 32 for c in url):
                            raise ValueError('Use an http or https article URL without credentials')
                        db.execute('INSERT INTO items(title,url) VALUES (?,?)', (title.strip(), url.strip()))
                    elif self.path == '/api/update':
                        if type(data.get('id')) is not int or type(data.get('is_read')) is not bool:
                            raise ValueError('Invalid article update')
                        result = db.execute('UPDATE items SET is_read=? WHERE id=?', (int(data['is_read']), data['id']))
                        if result.rowcount != 1:
                            self.send(404, {'error': 'Article not found'})
                            return
                    else:
                        self.send(404, {'error': 'Not found'})
                        return
                self.send(200, {'ok': True})
            except (ValueError, TypeError):
                self.send(400, {'error': 'Enter a title and valid http or https URL, or a valid article update.'})
            except sqlite3.Error:
                self.send(503, {'error': 'Storage is temporarily unavailable. Try again.'})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler), token


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start the local reading list. Python 3.11+, no packages.')
    parser.add_argument('--db', type=Path, default=Path.home()/'.usual/demo-reading-list.sqlite3')
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    server, token = make_server(args.db, args.port)
    print(f'Open this private link: http://127.0.0.1:{server.server_port}/#{token}', flush=True)
    print('Saved on this computer. Keep this terminal running; Ctrl+C stops it.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
