"""Private loopback review UI. A fresh in-memory capability protects every data request."""
import hmac
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

REVIEW_HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Usual · Review your run</title>
<style>*{box-sizing:border-box}body{background:#f5f3ed;color:#24392f;font:17px/1.6 system-ui;margin:0}main{max-width:1000px;padding:40px 24px;margin:auto}h1{font:48px/1.1 Georgia,serif;letter-spacing:-1.4px;margin:18px 0}h2{font-size:23px;line-height:1.3}p{margin:12px 0}.muted{color:#5b685f}.label{text-transform:uppercase;font-size:12px;letter-spacing:1.5px;font-weight:700}.card{border:1px solid #ced5c9;background:#fffefa;padding:26px;border-radius:14px;margin:22px 0}button{cursor:pointer;font:600 15px system-ui;padding:13px 18px;min-height:46px;border:1px solid #344f40;border-radius:7px;background:#344f40;color:white}.secondary{background:transparent;color:#344f40}button:disabled{opacity:.6;cursor:default}button:focus-visible,textarea:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid #ba613e;outline-offset:4px}.actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:18px}textarea{width:100%;padding:14px;font:16px/1.5 system-ui;min-height:90px;resize:vertical;margin:8px 0;border:1px solid #829480;border-radius:7px}blockquote{border-left:3px solid #b08a55;padding:8px 18px;margin:16px 0;white-space:pre-wrap;overflow-wrap:anywhere}.pill{display:inline-block;padding:4px 10px;background:#e7ecdf;border-radius:20px;font-size:13px}.error{color:#9e3826}.success{color:#24543b}.summary{display:flex;gap:15px;flex-wrap:wrap;padding:20px 0;border-block:1px solid #cbd4c5;margin:24px 0}.stat{flex:1;min-width:130px}.stat strong{font:36px Georgia;display:block}summary{cursor:pointer;padding:12px 0}a{color:#344f40}#message{min-height:28px}@media(max-width:600px){h1{font-size:38px}.card{padding:20px}main{padding:28px 18px}}</style></head>
<body><main><div class="label">Usual / private run review</div><h1>Your build kept moving.<br>Here’s the judgment behind it.</h1><p class="muted">Review the choices your agent made. Only choices you accept or correct become new evidence.</p><div id="message" role="status" aria-live="polite"></div><div id="content"></div></main>
<script>
'use strict';
const token=location.hash.slice(1);history.replaceState(null,'',location.pathname);const content=document.getElementById('content'),message=document.getElementById('message');
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function btn(label,fn,secondary=false){const b=el('button',label,secondary?'secondary':'');b.addEventListener('click',fn);return b;}
async function request(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});const data=await r.json();if(!r.ok)throw Error(data.error||'The request failed.');return data;}
async function review(id,verdict,statement,card){card.querySelectorAll('button').forEach(b=>b.disabled=true);try{await request('/api/review',{decision_id:id,verdict,statement});await load();message.textContent=verdict==='rejected'?'Rejected. This choice will not be learned.':'Saved. Your review will inform future consultations in this project.';message.className='success';}catch(e){message.textContent=e.message;message.className='error';card.querySelectorAll('button').forEach(b=>b.disabled=false);}}
async function load(){try{const report=await request('/api/report');content.replaceChildren();const run=report.run,s=report.summary;content.append(el('h2',run.task),el('p','Run '+run.id+' · '+run.status,'muted'));const stats=el('div',undefined,'summary');for(const [name,count] of [['Predictions',s.predictions],['Agent defaults',s.defaults],['Escalations',s.escalations],['Awaiting review',s.pending_review]]){const stat=el('div',undefined,'stat');stat.append(el('strong',count),el('span',name));stats.append(stat);}content.append(stats);content.append(el('p','Confidence is the agent’s own assessment, not a tested accuracy score. Review never grants permission to publish, spend, or perform an escalated action.','muted'));
for(const c of report.consultations){const card=el('article',undefined,'card'),d=c.decision;card.append(el('span',d?(d.basis.replaceAll('_',' ')+' · '+d.review):'Unresolved','pill'),el('h2',c.question));if(!d){card.append(el('p','This consultation still needs a recorded choice or escalation.'));content.append(card);continue;}card.append(el('strong',d.choice),el('p',d.rationale),el('p','Confidence: '+d.confidence,'muted'));if(d.correction)card.append(el('p','Your correction: '+d.correction,'success'));const details=el('details');details.append(el('summary','Evidence behind this choice ('+d.evidence_ids.length+')'));for(const evidence of c.evidence.filter(e=>d.evidence_ids.includes(e.id))){details.append(el('blockquote',evidence.quote),el('p',evidence.source+':'+evidence.line+' · '+evidence.origin,'muted'));}card.append(details);if(d.review==='pending'&&d.basis!=='escalated'){const actions=el('div',undefined,'actions');actions.append(btn('That’s my call',()=>review(d.id,'accepted','',card)),btn('I’d choose differently',()=>{form.hidden=!form.hidden;if(!form.hidden)field.focus();},true),btn('Don’t learn this',()=>review(d.id,'rejected','',card),true));const form=el('form');form.hidden=true;const label=el('label','What should Usual learn for this situation?');const field=el('textarea');field.maxLength=4000;field.required=true;field.id=d.id+'-correction';label.htmlFor=field.id;const submit=el('button','Save my correction');submit.type='submit';form.append(label,field,submit);form.addEventListener('submit',event=>{event.preventDefault();if(field.value.trim())review(d.id,'corrected',field.value.trim(),card);});card.append(actions,form);}else if(d.basis==='escalated'&&d.review==='pending'){card.append(el('p','This was left for you. Give any action permission directly in your coding session.'));card.append(btn('Acknowledge · don’t learn this',()=>review(d.id,'rejected','',card),true));}content.append(card);}
const download=btn('Download this run as JSON',()=>{const u=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));const a=el('a');a.href=u;a.download=run.id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);},true);content.append(download,el('p','Private to this computer and browser session. Stop the review server when you’re done. The database remains on your machine.','muted'));}catch(e){message.textContent=e.message;message.className='error';}}
if(token)load();else message.textContent='Open the complete private link printed by usual review-ui. Its access token is needed for this session.';
</script></body></html>'''


def make_server(store, run_id, port=0):
    store.report(run_id)  # Fail before opening a listener for an invalid run.
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Neither capabilities nor decision content enter access logs.

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
            super().end_headers()

        def reply(self, status, data):
            payload = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def allowed(self, auth=True):
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if host not in hosts or (origin and origin not in {f"http://{h}" for h in hosts}):
                self.reply(403, {"error": "This review is only available from its local origin."})
                return False
            if auth and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.reply(401, {"error": "Open the complete private review link printed in your terminal."})
                return False
            return True

        def do_GET(self):
            if not self.allowed(auth=urlparse(self.path).path != "/"):
                return
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(REVIEW_HTML.encode())
            elif self.path == "/api/report":
                self.reply(200, store.report(run_id))
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if not self.allowed():
                return
            if self.path != "/api/review":
                self.reply(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384 or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Invalid review request.")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Invalid review request.")
                allowed_ids = {c["decision"]["id"] for c in store.report(run_id)["consultations"] if c["decision"]}
                if data.get("decision_id") not in allowed_ids:
                    raise ValueError("This decision is not part of the current run.")
                result = store.review(data["decision_id"], data.get("verdict"), data.get("statement", ""))
                self.reply(200, result)
            except (ValueError, TypeError):
                self.reply(400, {"error": "Could not save this review. Check the correction and whether this choice is already reviewed."})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server, token


def serve(store, run_id, port=0):
    server, token = make_server(store, run_id, port)
    print(f"Private review: http://127.0.0.1:{server.server_port}/#{token}", flush=True)
    print("Keep this terminal running. Ctrl+C closes review access; your run stays saved.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
