// Shared sidebar nav + prev/next pager for the ESM2-L4 logbook.
// Single source of truth for the page list; each page just sets <body data-page="...">.
const PAGES = [
  {href:'index.html', label:'Overview',            short:'Overview',        status:null},
  {href:'m0.html',    label:'0 · Scaffold',        short:'Scaffold',        status:'done'},
  {href:'m1.html',    label:'1 · Environment',     short:'Environment',     status:'done'},
  {href:'m2.html',    label:'2 · Baseline',        short:'Baseline',        status:'done'},
  {href:'m3.html',    label:'3 · Correctness',     short:'Correctness',     status:'next'},
  {href:'m4.html',    label:'4 · Batching',        short:'Batching',        status:'todo'},
  {href:'m5.html',    label:'5 · torch.compile',   short:'torch.compile',   status:'todo'},
  {href:'m6.html',    label:'6 · Triton kernel',   short:'Triton kernel',   status:'todo'},
  {href:'m7.html',    label:'7 · Profiling',       short:'Profiling',       status:'todo'},
  {href:'m8.html',    label:'8 · Results & README',short:'Results & README',status:'todo'},
  {href:'m9.html',    label:'9 · Cleanup',         short:'Cleanup',         status:'todo'},
];

(function(){
  const current = document.body.dataset.page || 'index.html';
  const side = document.getElementById('side');
  if(side){
    let html =
      '<h1>ESM2 &times; L4</h1>'+
      '<div class="sub">Inference optimization logbook</div>'+
      '<div class="prog">2 / 9 milestones complete</div>';
    html += '<a href="index.html"'+(current==='index.html'?' class="active"':'')+'>'+
            '<span class="lbl">Overview</span></a>';
    html += '<div class="grp">Milestones</div>';
    for(const p of PAGES){
      if(p.href==='index.html') continue;
      const dot = p.status ? '<span class="dot '+p.status+'"></span>' : '';
      html += '<a href="'+p.href+'"'+(current===p.href?' class="active"':'')+'>'+
              dot+'<span class="lbl">'+p.label+'</span></a>';
    }
    side.innerHTML = html;
  }
  // prev/next pager
  const pager = document.getElementById('pager');
  if(pager){
    const i = PAGES.findIndex(p=>p.href===current);
    const prev = i>0 ? PAGES[i-1] : null;
    const next = i>=0 && i<PAGES.length-1 ? PAGES[i+1] : null;
    let h='';
    h += prev ? '<a class="prev" href="'+prev.href+'"><div class="dir">&larr; Previous</div><div class="t">'+prev.short+'</div></a>' : '<span></span>';
    h += next ? '<a class="next" href="'+next.href+'"><div class="dir">Next &rarr;</div><div class="t">'+next.short+'</div></a>' : '<span></span>';
    pager.innerHTML = h;
  }
})();
