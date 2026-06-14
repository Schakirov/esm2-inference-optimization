// Shared sidebar nav + prev/next pager for the ESM2-L4 logbook.
// Single source of truth for the page list; each page just sets <body data-page="...">.
const PAGES = [
  {href:'index.html', label:'Overview',            short:'Overview',        status:null},
  {href:'m0.html',    label:'0 · Scaffold',        short:'Scaffold',        status:'done'},
  {href:'m1.html',    label:'1 · Environment',     short:'Environment',     status:'done'},
  {href:'m2.html',    label:'2 · Baseline',        short:'Baseline',        status:'done'},
  {href:'m3.html',    label:'3 · Correctness',     short:'Correctness',     status:'done'},
  {href:'m4.html',    label:'4 · Batching',        short:'Batching',        status:'done'},
  {href:'m5.html',    label:'5 · torch.compile',   short:'torch.compile',   status:'done'},
  {href:'m6.html',    label:'6 · Triton kernel',   short:'Triton kernel',   status:'done'},
  {href:'m7.html',    label:'7 · Profiling',       short:'Profiling',       status:'done'},
  {href:'m8.html',    label:'8 · Results & README',short:'Results & README',status:'next'},
  {href:'m9.html',    label:'9 · Cleanup',         short:'Cleanup',         status:'todo'},
  {href:'kernel.html',label:'Triton kernel, annotated', short:'Annotated kernel', status:null, group:'deepdive'},
  {href:'hardware.html',label:'GPU & training context', short:'GPU context', status:null, group:'deepdive'},
  {href:'gemm.html',label:'Inside one GEMM kernel', short:'Inside one GEMM', status:null, group:'deepdive'},
];

(function(){
  const current = document.body.dataset.page || 'index.html';
  const side = document.getElementById('side');
  if(side){
    let html =
      '<h1>ESM2 &times; L4</h1>'+
      '<div class="sub">Inference optimization logbook</div>'+
      '<div class="prog">7 / 9 milestones complete</div>'+
      '<button class="themebtn" id="themebtn" type="button"></button>';
    html += '<a href="index.html"'+(current==='index.html'?' class="active"':'')+'>'+
            '<span class="lbl">Overview</span></a>';
    html += '<div class="grp">Milestones</div>';
    for(const p of PAGES){
      if(p.href==='index.html' || p.group) continue;
      const dot = p.status ? '<span class="dot '+p.status+'"></span>' : '';
      html += '<a href="'+p.href+'"'+(current===p.href?' class="active"':'')+'>'+
              dot+'<span class="lbl">'+p.label+'</span></a>';
    }
    const extras = PAGES.filter(p=>p.group==='deepdive');
    if(extras.length){
      html += '<div class="grp">Deep dive</div>';
      for(const p of extras){
        html += '<a href="'+p.href+'"'+(current===p.href?' class="active"':'')+'>'+
                '<span class="dot code"></span><span class="lbl">'+p.label+'</span></a>';
      }
    }
    side.innerHTML = html;

    // ---- dark/light theme toggle ----
    const KEY = 'esm2-theme';
    const btn = document.getElementById('themebtn');
    function currentTheme(){
      return document.documentElement.getAttribute('data-theme') || 'dark';
    }
    function paintBtn(){
      // Show the action: the theme you'd switch TO.
      const target = currentTheme() === 'light' ? 'dark' : 'light';
      const ico = target === 'dark' ? '🌙' : '☀';
      btn.innerHTML = '<span class="ico">'+ico+'</span><span>'+
        (target === 'dark' ? 'Dark mode' : 'Light mode')+'</span>';
    }
    function setTheme(t){
      if(t === 'dark'){ document.documentElement.removeAttribute('data-theme'); }
      else { document.documentElement.setAttribute('data-theme', t); }
      try { localStorage.setItem(KEY, t); } catch(e){}
      paintBtn();
    }
    if(btn){
      paintBtn();
      btn.addEventListener('click', ()=> setTheme(currentTheme() === 'light' ? 'dark' : 'light'));
    }
  }
  // prev/next pager (milestone sequence only; deep-dive pages are out of the flow)
  const pager = document.getElementById('pager');
  if(pager){
    const seq = PAGES.filter(p=>!p.group);
    const i = seq.findIndex(p=>p.href===current);
    const prev = i>0 ? seq[i-1] : null;
    const next = i>=0 && i<seq.length-1 ? seq[i+1] : null;
    let h='';
    h += prev ? '<a class="prev" href="'+prev.href+'"><div class="dir">&larr; Previous</div><div class="t">'+prev.short+'</div></a>' : '<span></span>';
    h += next ? '<a class="next" href="'+next.href+'"><div class="dir">Next &rarr;</div><div class="t">'+next.short+'</div></a>' : '<span></span>';
    pager.innerHTML = h;
  }
})();
