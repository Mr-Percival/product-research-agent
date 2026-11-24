"""Builds data/catalog/label_tool.html: a single-file tool to hand-label the 200 frozen test reviews.

The labeller sees only the review (never the model output), marks each aspect it discusses as
positive / negative / neutral, and exports labels as JSON (review_id -> {aspect: polarity}).
Progress autosaves in the browser. The HTML embeds review text, so it stays in data/ (git-ignored).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pra.taxonomy import ASPECTS

C = Path("data/catalog")

TEMPLATE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Etiquetado de reseñas</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1c1c1c;--mute:#666;--line:#ddd;--pos:#1f7a3a;--neg:#b3261e;--neu:#5f6b7a;--acc:#2b59c3}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
header b{font-size:18px}.bar{flex:1;min-width:160px;height:8px;background:var(--line);border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--acc)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:12px}
.meta{color:var(--mute);font-size:14px;margin-bottom:6px}.title{font-weight:600;margin-bottom:6px}
.text{white-space:pre-wrap;max-height:40vh;overflow:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:6px}
.asp{display:flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:8px;padding:6px 8px;background:var(--card)}
.asp span{flex:1;font-size:14px}.asp span small{display:block;color:var(--mute);font-size:12px;line-height:1.3}
.asp button{border:1px solid var(--line);background:#fff;border-radius:6px;width:34px;height:30px;font-weight:700;cursor:pointer}
.asp button.on[data-p=positive]{background:var(--pos);color:#fff;border-color:var(--pos)}
.asp button.on[data-p=negative]{background:var(--neg);color:#fff;border-color:var(--neg)}
.asp button.on[data-p=neutral]{background:var(--neu);color:#fff;border-color:var(--neu)}
.asp.set{border-color:var(--acc)}
nav{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}nav button{padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:#fff;cursor:pointer}
nav .primary{background:var(--acc);color:#fff;border-color:var(--acc)}.help{color:var(--mute);font-size:13px;margin-top:8px}
</style></head><body><main>
<header><b>Reseña <span id="n"></span> / __N__</b><div class="bar"><i id="prog"></i></div><span id="done"></span></header>
<div class="card"><div class="meta" id="meta"></div><div class="title" id="rt"></div><div class="text" id="tx"></div></div>
<div class="grid" id="grid"></div>
<nav><button id="prev">← Anterior</button><button class="primary" id="next">Guardar y siguiente →</button>
<button id="none">Sin aspectos (vacía)</button><button id="export">Exportar JSON</button></nav>
<p class="help">Marca solo los aspectos de los que la reseña habla: + positivo, − negativo, 0 neutral (hecho sin juicio).
Un clic de nuevo lo quita. "overall" solo si es satisfacción general sin aspecto. Flechas ← → para moverte. Se guarda solo en este navegador; al terminar, Exportar JSON.</p>
</main><script>
const R=__REVIEWS__, A=__ASPECTS__, KEY="pra_labels_v1";
let L={};try{L=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=0;try{i=+localStorage.getItem(KEY+"_i")||0}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(L));localStorage.setItem(KEY+"_i",i)}catch(e){}};
const P=[["positive","+"],["negative","−"],["neutral","0"]];
function render(){const r=R[i];document.getElementById("n").textContent=i+1;
document.getElementById("meta").textContent=r.category+" · "+r.rating+"★ · "+r.review_id.split(":")[0];
document.getElementById("rt").textContent=r.title;document.getElementById("tx").textContent=r.text;
const cur=(L[r.review_id]||{}).aspects||{};const g=document.getElementById("grid");g.innerHTML="";
for(const [k,d] of Object.entries(A)){const el=document.createElement("div");el.className="asp"+(cur[k]?" set":"");
el.innerHTML="<span>"+k+"<small>"+d+"</small></span>";
for(const [p,s] of P){const b=document.createElement("button");b.textContent=s;b.dataset.p=p;if(cur[k]===p)b.className="on";
b.onclick=()=>{const e=L[r.review_id]||{aspects:{}};if(e.aspects[k]===p)delete e.aspects[k];else e.aspects[k]=p;e.done=true;L[r.review_id]=e;save();render()};el.appendChild(b)}
g.appendChild(el)}
const d=Object.values(L).filter(x=>x.done).length;document.getElementById("done").textContent=d+" hechas";
document.getElementById("prog").style.width=(100*d/R.length)+"%"}
function go(s){i=Math.max(0,Math.min(R.length-1,i+s));save();render();scrollTo(0,0)}
document.getElementById("prev").onclick=()=>go(-1);
document.getElementById("next").onclick=()=>{const id=R[i].review_id;L[id]=L[id]||{aspects:{}};L[id].done=true;save();go(1)};
document.getElementById("none").onclick=()=>{L[R[i].review_id]={aspects:{},done:true};save();go(1)};
document.getElementById("export").onclick=()=>{const out={};for(const r of R)if(L[r.review_id]&&L[r.review_id].done)out[r.review_id]=L[r.review_id].aspects;
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:"application/json"}));
a.download="labels_"+Object.keys(out).length+".json";a.click()};
addEventListener("keydown",e=>{if(e.key==="ArrowRight")document.getElementById("next").click();if(e.key==="ArrowLeft")go(-1)});
render();
</script></body></html>"""


def main() -> None:
    label = pd.read_parquet(C / "label_sample.parquet").sample(frac=1, random_state=7)  # shuffle star order
    cat = pd.read_parquet(C / "products.parquet").set_index("parent_asin").category
    recs = [
        {
            "review_id": r.review_id,
            "category": cat[r.parent_asin],
            "rating": int(r.rating),
            "title": r.title,
            "text": r.text,
        }
        for r in label.itertuples()
    ]
    html = (
        TEMPLATE.replace("__REVIEWS__", json.dumps(recs).replace("</", "<\\/"))
        .replace("__ASPECTS__", json.dumps(ASPECTS))
        .replace("__N__", str(len(recs)))
    )
    (C / "label_tool.html").write_text(html, encoding="utf-8")
    print("written", C / "label_tool.html", len(recs), "reviews")


if __name__ == "__main__":
    main()
