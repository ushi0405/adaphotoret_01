const fs=require('fs'), vm=require('vm'), assert=require('assert/strict'), {webcrypto}=require('crypto');
const KEY='adaphotoret.photo-library.v1';
const SOURCE=fs.readFileSync(require('path').join(__dirname, '..', 'browser_library.js'),'utf8').replace('export default function(component)', 'globalThis.mount = function(component)');
const empty=()=>({photo_tags:{},libraries:{}}), album=name=>({photo_tags:{},libraries:{[name]:{photos:['data/travel/travel_23.HEIC'],tags:['tag']}}});
const wait=()=>new Promise(resolve=>setTimeout(resolve,20));
function harness(raw=null, options={}) {
 const db=new Map(raw===null?[]:[[KEY,raw]]), listeners=new Set(), events=[], writes=[];
 let queue=Promise.resolve();
 const localStorage={getItem(k){if(options.blocked)throw new Error('blocked');return db.get(k)??null},setItem(k,v){if(options.quota){let e=new Error('quota');e.name='QuotaExceededError';throw e}writes.push([k,v]);db.set(k,v)}};
 const context={TextEncoder,Uint8Array,crypto:webcrypto,console,localStorage,navigator:{locks:options.noLocks?undefined:{request(k,fn){let p=queue.then(fn);queue=p.catch(()=>{});return p}}},window:{addEventListener(type,fn){assert.equal(type,'storage');listeners.add(fn)},removeEventListener(type,fn){listeners.delete(fn)}}};
 vm.createContext(context);vm.runInContext(SOURCE,context);
 const mount=(request,parent={})=>({parent,dispose:context.mount({data:request,parentElement:parent,setStateValue(k,v){assert.equal(k,'event'); events.push(JSON.parse(JSON.stringify(v)))}})});
 const emitExternal=async()=>{for(const f of [...listeners])f({key:KEY});await wait()};
 return {db,events,writes,mount,emitExternal,listeners};
}
(async()=>{
 // Restore never writes; stale empty/default save cannot clobber restored data.
 const original=JSON.stringify({version:1,data:album('existing')}); let h=harness(original);
 h.mount({op:'load',id:'l'});await wait();assert.equal(h.events[0].kind,'loaded');assert.equal(h.writes.length,0);assert.deepEqual(h.events[0].data,album('existing'));
 h.mount({op:'save',id:'s',base_revision:null,data:empty()});await wait();assert.equal(h.events.at(-1).kind,'conflict');assert.equal(h.db.get(KEY),original);
 // Save acknowledgment follows actual serialized storage, interrupted request replay is idempotent.
 let rev=h.events[0].revision;h.mount({op:'save',id:'s2',base_revision:rev,data:album('new')});await wait();assert.equal(h.events.at(-1).kind,'saved');assert.deepEqual(JSON.parse(h.db.get(KEY)).data,album('new'));
 const n=h.writes.length;h.mount({op:'save',id:'s2',base_revision:rev,data:album('new')});await wait();assert.equal(h.writes.length,n);assert.equal(h.events.at(-1).kind,'saved');
 // Delete is persisted, then restored into a new component lifecycle.
 rev=h.events.at(-1).revision;h.mount({op:'save',id:'del',base_revision:rev,data:empty()});await wait();h.mount({op:'load',id:'refresh'});await wait();assert.deepEqual(h.events.at(-1).data,empty());
 // Concurrent stale writers serialize and reject the loser, never resurrecting discarded content.
 h=harness();h.mount({op:'save',id:'a',base_revision:null,data:album('A')});h.mount({op:'save',id:'b',base_revision:null,data:album('B')});await wait();assert.deepEqual(h.events.map(e=>e.kind),['saved','conflict']);assert.deepEqual(JSON.parse(h.db.get(KEY)).data,album('A'));
 // Blocked storage, corrupt data, quota, and missing Locks fail visibly with original bytes intact.
 for(const [opts,raw,request,code] of [[{blocked:true},null,{op:'load',id:'blocked'},'unavailable'],[{},'{bad',{op:'load',id:'corrupt'},'invalid'],[{quota:true},null,{op:'save',id:'quota',base_revision:null,data:album('A')},'quota'],[{noLocks:true},null,{op:'save',id:'unsupported',base_revision:null,data:album('A')},'unsupported']]) {
   h=harness(raw,opts);h.mount(request);await wait();assert.equal(h.events.at(-1).kind,'error');assert.equal(h.events.at(-1).code,code);assert.equal(h.db.get(KEY)??null,raw);assert.equal(h.writes.length,0);
 }
 // Other-tab updates notify watchers, and cleanup actually unregisters.
 h=harness();const one=h.mount({op:'watch',id:'watch'});h.db.set(KEY,original);await h.emitExternal();assert.equal(h.events.at(-1).kind,'external');assert.deepEqual(h.events.at(-1).data,album('existing'));one.dispose();assert.equal(h.listeners.size,0);
 // A save/load acknowledgment cannot be coalesced away by storage notifications.
 h=harness();const parent={};const saving=h.mount({op:'save',id:'coalesce',base_revision:null,data:album('A')},parent);
 await wait();assert.equal(h.events.at(-1).kind,'saved');assert.equal(h.listeners.size,0);
 h.db.set(KEY,original);await h.emitExternal();assert.equal(h.events.at(-1).kind,'saved');
 saving.dispose();h.mount({op:'watch',id:'watch-after-ack'},parent);await wait();
 assert.equal(h.events.at(-1).kind,'external');assert.equal(h.events.at(-1).request_id,'watch-after-ack');assert.deepEqual(h.events.at(-1).data,album('existing'));
 // Malicious-looking text is inert data, not markup execution or property insertion.
 h=harness();let bad=album('<img src=x onerror=alert(1)>');h.mount({op:'save',id:'xss',base_revision:null,data:bad});await wait();assert.deepEqual(JSON.parse(h.db.get(KEY)).data,bad);assert.equal(h.events.at(-1).kind,'saved');
 console.log('PASS: 13 independent Node VM scenario groups: restore/no-write, stale-save conflict, acked save, idempotent replay, delete+restore, concurrent stale writers, blocked storage, corrupt storage, quota, unsupported locks, external update+cleanup, inert untrusted text. This is a JS protocol simulation, NOT a real browser test.');
})().catch(e=>{console.error(e);process.exit(1)});
