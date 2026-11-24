// Both adapters receive identical undirected edits and return the same row-patch format.
import readline from 'node:readline'
import { D2, map, count, innerJoin, consolidate, output, MessageType } from '@electric-sql/d2ts'

const mode = process.argv[2]
let version = 0, input, graph, adjacency, weighted, dirty
function init(n) {
  weighted = Array.from({length:n}, () => new Map())
  if (mode === 'd2ts') {
    graph = new D2({initialFrontier:0})
    input = graph.newInput()
    const degrees = input.pipe(count())
    input.pipe(innerJoin(degrees), map(([src,[dst,ds]]) => [dst,[src,ds]]),
      innerJoin(degrees), map(([dst,[[src,ds],dt]]) => [dst,src,ds*dt]),
      consolidate(), output(msg => {
        if (msg.type !== MessageType.DATA) return
        const changes = msg.data.collection.getInner()
        for (const sign of [-1,1]) for (const [[dst,src,product],m] of changes) {
          if (Math.sign(m) !== sign) continue
          if (sign < 0) weighted[dst].delete(src)
          else weighted[dst].set(src,product)
          dirty.add(dst)
        }
      }))
    graph.finalize()
  } else {
    adjacency = Array.from({length:n}, () => new Set())
  }
}
function handle(req) {
  if (req.n) init(req.n)
  dirty = new Set()
  if (mode === 'd2ts') {
    const arcs = req.edits.flatMap(([a,b,m]) => [[[a,b],m],[[b,a],m]])
    if (arcs.length) input.sendData(version,arcs)
    input.sendFrontier(++version)
    graph.run()
  } else {
    const ends = new Set()
    for (const [a,b,m] of req.edits) {
      for (const [u,v] of [[a,b],[b,a]]) {
        if (m === 1) adjacency[u].add(v)
        else adjacency[u].delete(v)
        ends.add(u)
      }
    }
    for (const u of ends) {
      dirty.add(u)
      for (const v of adjacency[u]) dirty.add(v)
    }
    for (const u of dirty) {
      weighted[u].clear()
      for (const v of adjacency[u]) weighted[u].set(v,adjacency[u].size*adjacency[v].size)
    }
  }
  if (req.n) return {ready:true}
  for (const u of req.features) for (const v of weighted[u].keys()) dirty.add(v)
  return Array.from(dirty, u => [u,Array.from(weighted[u]).sort((a,b)=>a[0]-b[0])])
}
const lines = readline.createInterface({input:process.stdin})
for await (const line of lines) process.stdout.write(JSON.stringify(handle(JSON.parse(line)))+'\n')
