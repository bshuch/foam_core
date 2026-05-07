import React, { useState, useRef, useEffect, useMemo, useCallback } from "react";
import * as THREE from "three";

// ============================================================
// FOAM CORE MOCKUP SIMULATOR
// ------------------------------------------------------------
// 2D mode: draw outline polygon (rect default), place fold lines
//   (blue) and cut lines (red). Each line is a segment defined
//   by two points that must touch the outline boundary or
//   another existing line so the sheet partitions into faces.
// 3D mode: build a planar graph of faces from outline + lines,
//   walk from a root face along non-cut fold edges, applying
//   the per-edge rotation (mountain/valley × angle).
// ============================================================

// ---------- Geometry helpers ----------
const EPS = 1e-6;

const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
const add = (a, b) => ({ x: a.x + b.x, y: a.y + b.y });
const scale = (a, s) => ({ x: a.x * s, y: a.y * s });
const dot = (a, b) => a.x * b.x + a.y * b.y;
const cross2 = (a, b) => a.x * b.y - a.y * b.x;
const len = (a) => Math.hypot(a.x, a.y);
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function pointOnSegment(p, a, b) {
  const ab = sub(b, a);
  const ap = sub(p, a);
  const c = cross2(ab, ap);
  if (Math.abs(c) > 0.5) return false; // pixel-ish tolerance
  const d = dot(ab, ap);
  const l2 = dot(ab, ab);
  return d >= -EPS && d <= l2 + EPS;
}

function segIntersect(p1, p2, p3, p4) {
  // returns intersection point if segments properly intersect (not at endpoints), else null
  const r = sub(p2, p1);
  const s = sub(p4, p3);
  const rxs = cross2(r, s);
  if (Math.abs(rxs) < EPS) return null;
  const qp = sub(p3, p1);
  const t = cross2(qp, s) / rxs;
  const u = cross2(qp, r) / rxs;
  if (t > EPS && t < 1 - EPS && u > EPS && u < 1 - EPS) {
    return { x: p1.x + t * r.x, y: p1.y + t * r.y };
  }
  return null;
}

function pointInPolygon(p, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x,
      yi = poly[i].y;
    const xj = poly[j].x,
      yj = poly[j].y;
    const intersect =
      yi > p.y !== yj > p.y &&
      p.x < ((xj - xi) * (p.y - yi)) / (yj - yi + EPS) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function polygonArea(poly) {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const p1 = poly[i];
    const p2 = poly[(i + 1) % poly.length];
    a += p1.x * p2.y - p2.x * p1.y;
  }
  return a / 2;
}

function polygonCentroid(poly) {
  let cx = 0,
    cy = 0,
    a = 0;
  for (let i = 0; i < poly.length; i++) {
    const p1 = poly[i];
    const p2 = poly[(i + 1) % poly.length];
    const f = p1.x * p2.y - p2.x * p1.y;
    a += f;
    cx += (p1.x + p2.x) * f;
    cy += (p1.y + p2.y) * f;
  }
  a *= 0.5;
  if (Math.abs(a) < EPS) {
    // fallback: average
    const n = poly.length;
    return {
      x: poly.reduce((s, p) => s + p.x, 0) / n,
      y: poly.reduce((s, p) => s + p.y, 0) / n,
    };
  }
  return { x: cx / (6 * a), y: cy / (6 * a) };
}

// ---------- Planar subdivision ----------
// Build vertices, edges, faces from outline polygon + interior line segments.
// Lines must have endpoints on outline or on each other (we snap during edit).
const SNAP_TOL = 1.5; // pixels — tolerance for "same vertex"
const ON_SEG_TOL = 1.5; // pixels — perpendicular distance for "on segment"

function buildPlanarGraph(outline, lines) {
  // Collect all segments: outline edges + line segments
  const segs = [];
  for (let i = 0; i < outline.length; i++) {
    segs.push({
      a: { x: outline[i].x, y: outline[i].y },
      b: { x: outline[(i + 1) % outline.length].x, y: outline[(i + 1) % outline.length].y },
      type: "outline",
      lineId: null,
    });
  }
  for (const ln of lines) {
    segs.push({
      a: { x: ln.a.x, y: ln.a.y },
      b: { x: ln.b.x, y: ln.b.y },
      type: ln.type,
      lineId: ln.id,
    });
  }
  // Drop degenerate segments
  const validSegs = segs.filter(s => dist(s.a, s.b) > SNAP_TOL);

  // Robust vertex deduplication: linear scan with distance threshold.
  // (For the modest vertex counts here, O(n²) is fine and avoids all the
  // grid-rounding ambiguity that bit us.)
  const verts = [];
  function addVert(p) {
    for (let i = 0; i < verts.length; i++) {
      if (dist(verts[i], p) <= SNAP_TOL) return i;
    }
    verts.push({ x: p.x, y: p.y });
    return verts.length - 1;
  }

  // For each segment compute parameter t for each split point
  const segPoints = validSegs.map((s) => [
    { p: s.a, t: 0 },
    { p: s.b, t: 1 },
  ]);

  // Helper: distance from point to segment
  function pointSegDist(p, a, b) {
    const ab = sub(b, a);
    const l2 = dot(ab, ab);
    if (l2 < EPS) return dist(p, a);
    const t = Math.max(0, Math.min(1, dot(sub(p, a), ab) / l2));
    const proj = add(a, scale(ab, t));
    return dist(p, proj);
  }

  // Proper segment intersection using parameter form
  function segIntersectParam(p1, p2, p3, p4) {
    const r = sub(p2, p1);
    const s = sub(p4, p3);
    const rxs = cross2(r, s);
    if (Math.abs(rxs) < EPS) return null;
    const qp = sub(p3, p1);
    const t = cross2(qp, s) / rxs;
    const u = cross2(qp, r) / rxs;
    // Allow intersection points slightly inside (we'll let endpoints coincide
    // separately via dedup)
    const margin = SNAP_TOL / Math.max(len(r), len(s));
    if (t > margin && t < 1 - margin && u > margin && u < 1 - margin) {
      return { x: p1.x + t * r.x, y: p1.y + t * r.y, t, u };
    }
    return null;
  }

  for (let i = 0; i < validSegs.length; i++) {
    for (let j = i + 1; j < validSegs.length; j++) {
      const ip = segIntersectParam(validSegs[i].a, validSegs[i].b, validSegs[j].a, validSegs[j].b);
      if (ip) {
        segPoints[i].push({ p: { x: ip.x, y: ip.y }, t: ip.t });
        segPoints[j].push({ p: { x: ip.x, y: ip.y }, t: ip.u });
      }
      // T-junctions: endpoint of one segment on interior of another
      const segI = validSegs[i], segJ = validSegs[j];
      const lenI = dist(segI.a, segI.b);
      const lenJ = dist(segJ.a, segJ.b);
      for (const ep of [segJ.a, segJ.b]) {
        if (pointSegDist(ep, segI.a, segI.b) <= ON_SEG_TOL) {
          const t = dot(sub(ep, segI.a), sub(segI.b, segI.a)) / (lenI * lenI);
          if (t > SNAP_TOL / lenI && t < 1 - SNAP_TOL / lenI) {
            segPoints[i].push({ p: { x: ep.x, y: ep.y }, t });
          }
        }
      }
      for (const ep of [segI.a, segI.b]) {
        if (pointSegDist(ep, segJ.a, segJ.b) <= ON_SEG_TOL) {
          const t = dot(sub(ep, segJ.a), sub(segJ.b, segJ.a)) / (lenJ * lenJ);
          if (t > SNAP_TOL / lenJ && t < 1 - SNAP_TOL / lenJ) {
            segPoints[j].push({ p: { x: ep.x, y: ep.y }, t });
          }
        }
      }
    }
  }

  // Build edges (deduped) with type info
  const edgeMap = new Map(); // "u-v" sorted -> {u,v,type,lineId}
  const edgeKey = (u, v) => (u < v ? `${u}-${v}` : `${v}-${u}`);

  for (let i = 0; i < validSegs.length; i++) {
    const pts = segPoints[i].sort((a, b) => a.t - b.t);
    // dedupe by t (close t values)
    const dedup = [];
    for (const pt of pts) {
      if (dedup.length === 0 || pt.t - dedup[dedup.length - 1].t > 1e-4) {
        dedup.push(pt);
      }
    }
    for (let k = 0; k < dedup.length - 1; k++) {
      const u = addVert(dedup[k].p);
      const v = addVert(dedup[k + 1].p);
      if (u === v) continue;
      const key = edgeKey(u, v);
      const existing = edgeMap.get(key);
      // priority: cut > fold > outline (if same edge has multiple roles)
      const priority = { outline: 0, fold: 1, cut: 2 };
      if (!existing || priority[validSegs[i].type] > priority[existing.type]) {
        edgeMap.set(key, {
          u,
          v,
          type: validSegs[i].type,
          lineId: validSegs[i].lineId,
        });
      }
    }
  }

  const edges = Array.from(edgeMap.values());

  // Build adjacency: for each vertex, list of (neighbor, edgeIdx, angle)
  const adj = verts.map(() => []);
  edges.forEach((e, ei) => {
    const a1 = Math.atan2(verts[e.v].y - verts[e.u].y, verts[e.v].x - verts[e.u].x);
    const a2 = Math.atan2(verts[e.u].y - verts[e.v].y, verts[e.u].x - verts[e.v].x);
    adj[e.u].push({ to: e.v, edge: ei, angle: a1 });
    adj[e.v].push({ to: e.u, edge: ei, angle: a2 });
  });

  // Sort each adjacency by angle
  adj.forEach((list) => list.sort((a, b) => a.angle - b.angle));

  // Face traversal: for each directed half-edge, walk by always taking
  // the next edge clockwise (i.e., the previous in CCW order) at the
  // arrival vertex, until returning to start.
  const visited = new Set(); // "u->v"
  const faces = [];

  function nextHalfEdge(u, v) {
    // arrived at v from u; pick the half-edge leaving v that is the
    // most clockwise turn from the incoming direction.
    const incomingAngle = Math.atan2(verts[u].y - verts[v].y, verts[u].x - verts[v].x);
    const list = adj[v];
    // find index of (to=u) in list
    let idx = list.findIndex((e) => e.to === u);
    // The next edge clockwise around v is at index (idx - 1) mod n
    const n = list.length;
    const nextIdx = (idx - 1 + n) % n;
    return list[nextIdx];
  }

  for (let u = 0; u < verts.length; u++) {
    for (const { to: v, edge: ei } of adj[u]) {
      const key = `${u}->${v}`;
      if (visited.has(key)) continue;
      // walk
      const faceVerts = [u];
      const faceEdges = [];
      let cu = u,
        cv = v;
      let safety = 0;
      let closed = false;
      while (safety++ < 1000) {
        visited.add(`${cu}->${cv}`);
        faceVerts.push(cv);
        const incomingEdge = adj[cu].find((e) => e.to === cv);
        if (!incomingEdge) break;
        faceEdges.push(incomingEdge.edge);
        const nxt = nextHalfEdge(cu, cv);
        if (!nxt) break;
        cu = cv;
        cv = nxt.to;
        if (cu === u && cv === v) { closed = true; break; }
      }
      if (!closed) continue; // bad walk: don't store
      faceVerts.pop(); // last is duplicate of first start
      if (faceVerts.length < 3) continue;
      // compute area; outer face has negative area (CW with our convention)
      const poly = faceVerts.map((i) => verts[i]);
      const area = polygonArea(poly);
      if (area > 1) { // require meaningful area, not just > EPS
        faces.push({ verts: faceVerts, edges: faceEdges, area, poly });
      }
    }
  }

  // Filter: keep only faces whose centroid is inside the outline
  const validFaces = faces.filter((f) => {
    const c = polygonCentroid(f.poly);
    return pointInPolygon(c, outline);
  });

  // Build face adjacency via shared fold edges
  const faceAdj = validFaces.map(() => []);
  const edgeToFaces = new Map();
  validFaces.forEach((f, fi) => {
    f.edges.forEach((ei) => {
      if (!edgeToFaces.has(ei)) edgeToFaces.set(ei, []);
      edgeToFaces.get(ei).push(fi);
    });
  });
  for (const [ei, fs] of edgeToFaces.entries()) {
    if (fs.length === 2) {
      const e = edges[ei];
      faceAdj[fs[0]].push({ face: fs[1], edge: ei, edgeData: e });
      faceAdj[fs[1]].push({ face: fs[0], edge: ei, edgeData: e });
    }
  }

  // Handle interior flaps: a fold edge with only ONE adjacent face means
  // the other "face" is actually the larger face that contains it as a
  // hole. The standard planar walker doesn't handle holes, so we connect
  // the orphan face to whichever larger face geometrically contains it.
  for (const [ei, fs] of edgeToFaces.entries()) {
    if (fs.length === 1) {
      const e = edges[ei];
      if (e.type !== "fold") continue; // only fold edges need bridging
      const orphan = fs[0];
      const orphanCentroid = polygonCentroid(validFaces[orphan].poly);
      // find smallest face that contains the orphan's centroid (excluding self)
      let host = -1;
      let hostArea = Infinity;
      for (let fi = 0; fi < validFaces.length; fi++) {
        if (fi === orphan) continue;
        if (pointInPolygon(orphanCentroid, validFaces[fi].poly) &&
            validFaces[fi].area > validFaces[orphan].area &&
            validFaces[fi].area < hostArea) {
          host = fi;
          hostArea = validFaces[fi].area;
        }
      }
      if (host !== -1) {
        faceAdj[orphan].push({ face: host, edge: ei, edgeData: e });
        faceAdj[host].push({ face: orphan, edge: ei, edgeData: e });
      }
    }
  }

  return { verts, edges, faces: validFaces, faceAdj };
}

// ---------- 3D folding ----------
// Walk faces from root via fold edges (skip cut edges), applying per-edge
// rotation. Each face gets a 4x4 transform.
function computeFolded(graph, lineProps) {
  const { verts, edges, faces, faceAdj } = graph;
  if (faces.length === 0) return { faces: [], transforms: [] };

  // Choose root: largest face
  let root = 0;
  for (let i = 1; i < faces.length; i++)
    if (faces[i].area > faces[root].area) root = i;

  const transforms = faces.map(() => null);
  transforms[root] = new THREE.Matrix4().identity();

  const queue = [root];
  const visited = new Set([root]);

  while (queue.length) {
    const fi = queue.shift();
    const Tparent = transforms[fi];
    for (const { face: nf, edge: ei, edgeData } of faceAdj[fi]) {
      if (visited.has(nf)) continue;
      // Only traverse via fold edges; cut edges block; outline edges (degree 1) won't appear here
      if (edgeData.type !== "fold") continue;
      // Get fold parameters
      const lp = lineProps[edgeData.lineId] || { angle: 90, direction: "valley" };
      // Mountain folds bend down (negative), valley folds bend up (positive)
      // Angle is the deviation from flat (180°). We rotate by `angle` degrees.
      const sign = lp.direction === "mountain" ? -1 : 1;
      const angleRad = sign * (lp.angle * Math.PI) / 180;

      // Edge in 2D
      const a2 = verts[edgeData.u];
      const b2 = verts[edgeData.v];
      // Transform edge points by parent transform first to get current 3D positions
      const aP = new THREE.Vector3(a2.x, a2.y, 0).applyMatrix4(Tparent);
      const bP = new THREE.Vector3(b2.x, b2.y, 0).applyMatrix4(Tparent);
      const axis = new THREE.Vector3().subVectors(bP, aP).normalize();

      // To know fold orientation consistently, we want to rotate the child
      // face about the edge so it bends out of the parent's plane. We need
      // a consistent "side" choice: the rotation direction must be the same
      // regardless of which face is parent. We use a canonical direction:
      // determine which side of the edge the child face's centroid lies on
      // in the parent's local frame, and rotate accordingly.

      // Get child face centroid in parent's local coordinates (the original 2D)
      const childCentroid2D = polygonCentroid(faces[nf].poly);
      // edge direction in 2D
      const ex = b2.x - a2.x;
      const ey = b2.y - a2.y;
      // perpendicular from edge to child centroid (2D cross sign)
      const cx = childCentroid2D.x - a2.x;
      const cy = childCentroid2D.y - a2.y;
      const sideSign = Math.sign(ex * cy - ey * cx); // +1 or -1

      // Build rotation about world-space axis through aP
      // Apply: T_child = Translate(aP) * Rot(axis, angle*sideSign) * Translate(-aP) * Tparent
      const rotation = new THREE.Matrix4().makeRotationAxis(
        axis,
        angleRad * sideSign
      );
      const toOrigin = new THREE.Matrix4().makeTranslation(-aP.x, -aP.y, -aP.z);
      const fromOrigin = new THREE.Matrix4().makeTranslation(aP.x, aP.y, aP.z);
      const Tchild = new THREE.Matrix4()
        .multiply(fromOrigin)
        .multiply(rotation)
        .multiply(toOrigin)
        .multiply(Tparent);

      transforms[nf] = Tchild;
      visited.add(nf);
      queue.push(nf);
    }
  }

  return { faces, transforms, verts, root };
}

// ---------- Triangulate a simple polygon (ear clipping) ----------
function triangulate(poly) {
  const verts = poly.map((p, i) => i);
  const tris = [];
  if (poly.length < 3) return tris;
  // Ensure CCW
  let area = polygonArea(poly);
  const idx = area > 0 ? verts.slice() : verts.slice().reverse();
  let safety = 0;
  while (idx.length > 3 && safety++ < 1000) {
    let earFound = false;
    for (let i = 0; i < idx.length; i++) {
      const i0 = idx[(i - 1 + idx.length) % idx.length];
      const i1 = idx[i];
      const i2 = idx[(i + 1) % idx.length];
      const a = poly[i0],
        b = poly[i1],
        c = poly[i2];
      const cr = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
      if (cr <= 0) continue;
      // check no other vertex inside this triangle
      let ok = true;
      for (let k = 0; k < idx.length; k++) {
        if (k === (i - 1 + idx.length) % idx.length || k === i || k === (i + 1) % idx.length) continue;
        const p = poly[idx[k]];
        // barycentric inside test
        const d1 = (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
        const d2 = (p.x - b.x) * (c.y - b.y) - (p.y - b.y) * (c.x - b.x);
        const d3 = (p.x - c.x) * (a.y - c.y) - (p.y - c.y) * (a.x - c.x);
        const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
        const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
        if (!(hasNeg && hasPos)) {
          ok = false;
          break;
        }
      }
      if (ok) {
        tris.push([i0, i1, i2]);
        idx.splice(i, 1);
        earFound = true;
        break;
      }
    }
    if (!earFound) break;
  }
  if (idx.length === 3) tris.push([idx[0], idx[1], idx[2]]);
  return tris;
}

// ============================================================
// COMPONENTS
// ============================================================

function Editor2D({
  outline, setOutline,
  lines, setLines,
  lineProps, setLineProps,
  selectedLineId, setSelectedLineId,
  selectedFlapId, setSelectedFlapId,
  tool, setTool,
  deleteSelected,
}) {
  const svgRef = useRef(null);
  const [drawStart, setDrawStart] = useState(null);
  const [hoverPt, setHoverPt] = useState(null);
  const [draggingVertex, setDraggingVertex] = useState(null);
  // For flap tool: rectangle drag state
  const [flapStart, setFlapStart] = useState(null);
  const [flapCurrent, setFlapCurrent] = useState(null);

  const W = 800, H = 600;
  const padding = 40;

  // Snap a point to outline edges, existing line endpoints, or existing line segments
  const snap = useCallback((p) => {
    let best = null;
    let bestD = 8;
    // outline vertices
    for (const v of outline) {
      const d = dist(p, v);
      if (d < bestD) { bestD = d; best = { ...v }; }
    }
    // line endpoints
    for (const ln of lines) {
      for (const ep of [ln.a, ln.b]) {
        const d = dist(p, ep);
        if (d < bestD) { bestD = d; best = { ...ep }; }
      }
    }
    // outline edges (project)
    bestD = 10;
    if (!best) {
      for (let i = 0; i < outline.length; i++) {
        const a = outline[i], b = outline[(i+1)%outline.length];
        const ab = sub(b, a);
        const t = Math.max(0, Math.min(1, dot(sub(p, a), ab) / dot(ab, ab)));
        const proj = add(a, scale(ab, t));
        const d = dist(p, proj);
        if (d < bestD) { bestD = d; best = proj; }
      }
      // line segments
      for (const ln of lines) {
        const a = ln.a, b = ln.b;
        const ab = sub(b, a);
        const t = Math.max(0, Math.min(1, dot(sub(p, a), ab) / dot(ab, ab)));
        const proj = add(a, scale(ab, t));
        const d = dist(p, proj);
        if (d < bestD) { bestD = d; best = proj; }
      }
    }
    return best || p;
  }, [outline, lines]);

  function svgCoords(e) {
    const svg = svgRef.current;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const ctm = svg.getScreenCTM().inverse();
    const t = pt.matrixTransform(ctm);
    return { x: t.x, y: t.y };
  }

  function handleMouseDown(e) {
    const p = svgCoords(e);
    if (tool === "edit") {
      // try grabbing an outline vertex
      for (let i = 0; i < outline.length; i++) {
        if (dist(p, outline[i]) < 10) {
          setDraggingVertex(i);
          return;
        }
      }
    } else if (tool === "fold" || tool === "cut") {
      const sp = snap(p);
      setDrawStart(sp);
    } else if (tool === "flap") {
      // Drag a rectangle entirely inside the outline
      if (pointInPolygon(p, outline)) {
        setFlapStart(p);
        setFlapCurrent(p);
      }
    }
  }

  function handleMouseMove(e) {
    const p = svgCoords(e);
    if (draggingVertex !== null) {
      const newOutline = outline.slice();
      newOutline[draggingVertex] = { x: Math.max(padding, Math.min(W-padding, p.x)), y: Math.max(padding, Math.min(H-padding, p.y)) };
      setOutline(newOutline);
      return;
    }
    if (flapStart) {
      setFlapCurrent(p);
      return;
    }
    if (drawStart) {
      setHoverPt(snap(p));
    } else {
      setHoverPt(null);
    }
  }

  function handleMouseUp(e) {
    if (draggingVertex !== null) {
      setDraggingVertex(null);
      return;
    }
    if (flapStart && flapCurrent) {
      // Build the flap rectangle: 4 corners, 1 fold (hinge) + 3 cuts
      const x1 = Math.min(flapStart.x, flapCurrent.x);
      const x2 = Math.max(flapStart.x, flapCurrent.x);
      const y1 = Math.min(flapStart.y, flapCurrent.y);
      const y2 = Math.max(flapStart.y, flapCurrent.y);
      const w = x2 - x1, h = y2 - y1;
      // Validate: minimum size and entirely inside outline
      const corners = [
        { x: x1, y: y1 }, { x: x2, y: y1 },
        { x: x2, y: y2 }, { x: x1, y: y2 },
      ];
      const allInside = corners.every(c => pointInPolygon(c, outline));
      if (w >= 12 && h >= 12 && allInside) {
        const flapId = `flap_${Date.now()}_${Math.random().toString(36).slice(2,7)}`;
        // Default hinge: bottom edge (between corner 3 and corner 2 → bottom)
        // Edges: 0=top(c0-c1), 1=right(c1-c2), 2=bottom(c2-c3), 3=left(c3-c0)
        const hinge = "bottom"; // user can switch later
        const edgeMap = {
          top:    { a: corners[0], b: corners[1] },
          right:  { a: corners[1], b: corners[2] },
          bottom: { a: corners[3], b: corners[2] },
          left:   { a: corners[0], b: corners[3] },
        };
        const sides = ["top", "right", "bottom", "left"];
        const newLines = [];
        const newProps = { ...lineProps };
        for (const side of sides) {
          const id = `${flapId}_${side}`;
          const { a, b } = edgeMap[side];
          const isHinge = side === hinge;
          newLines.push({
            id, a, b,
            type: isHinge ? "fold" : "cut",
            flapId, flapSide: side,
          });
          if (isHinge) newProps[id] = { angle: 90, direction: "valley" };
        }
        setLines([...lines, ...newLines]);
        setLineProps(newProps);
        setSelectedFlapId(flapId);
        setSelectedLineId(null);
      }
      setFlapStart(null);
      setFlapCurrent(null);
      return;
    }
    if (drawStart) {
      const p = svgCoords(e);
      const sp = snap(p);
      if (dist(drawStart, sp) > 8) {
        const id = `ln_${Date.now()}_${Math.random().toString(36).slice(2,7)}`;
        setLines([...lines, { id, a: drawStart, b: sp, type: tool }]);
        if (tool === "fold") {
          setLineProps({
            ...lineProps,
            [id]: { angle: 90, direction: "valley" }
          });
        }
        setSelectedLineId(id);
        setSelectedFlapId(null);
      }
      setDrawStart(null);
      setHoverPt(null);
    }
  }

  useEffect(() => {
    const onKey = (e) => {
      if ((e.key === "Delete" || e.key === "Backspace") && (selectedLineId || selectedFlapId)) {
        const tag = document.activeElement?.tagName;
        if (tag !== "INPUT" && tag !== "TEXTAREA") {
          e.preventDefault();
          deleteSelected();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedLineId, selectedFlapId, deleteSelected]);

  const outlinePath = useMemo(() => {
    if (outline.length === 0) return "";
    return "M " + outline.map(p => `${p.x},${p.y}`).join(" L ") + " Z";
  }, [outline]);

  return (
    <div style={styles.canvasWrap}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        style={styles.canvas}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => { setDrawStart(null); setHoverPt(null); setDraggingVertex(null); setFlapStart(null); setFlapCurrent(null); }}
      >
        {/* Drafting grid */}
        <defs>
          <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#d4c9a8" strokeWidth="0.5"/>
          </pattern>
          <pattern id="grid-major" width="100" height="100" patternUnits="userSpaceOnUse">
            <rect width="100" height="100" fill="url(#grid)"/>
            <path d="M 100 0 L 0 0 0 100" fill="none" stroke="#b8a880" strokeWidth="0.8"/>
          </pattern>
        </defs>
        <rect width={W} height={H} fill="url(#grid-major)"/>

        {/* Outline fill */}
        <path d={outlinePath} fill="#f4ecd6" fillOpacity="0.85" stroke="#3a3528" strokeWidth="1.5"/>

        {/* Lines */}
        {lines.map(ln => {
          const isSel = ln.id === selectedLineId;
          const isFlapSel = ln.flapId && ln.flapId === selectedFlapId;
          const highlight = isSel || isFlapSel;
          const color = ln.type === "fold" ? "#2563a6" : "#c0392b";
          const dash = ln.type === "fold" ? "8 4" : "0";
          return (
            <g key={ln.id} onMouseDown={(e) => {
              if (!e.shiftKey) {
                e.stopPropagation();
                if (ln.flapId) {
                  setSelectedFlapId(ln.flapId);
                  setSelectedLineId(null);
                } else {
                  setSelectedLineId(ln.id);
                  setSelectedFlapId(null);
                }
              }
            }} style={{ cursor: "pointer" }}>
              <line x1={ln.a.x} y1={ln.a.y} x2={ln.b.x} y2={ln.b.y}
                stroke="transparent" strokeWidth="14" strokeLinecap="round"/>
              <line x1={ln.a.x} y1={ln.a.y} x2={ln.b.x} y2={ln.b.y}
                stroke={color} strokeWidth={highlight ? 4 : 2.5} strokeDasharray={dash} strokeLinecap="round"
                pointerEvents="none"/>
              {highlight && (
                <line x1={ln.a.x} y1={ln.a.y} x2={ln.b.x} y2={ln.b.y}
                  stroke={color} strokeWidth="10" strokeOpacity="0.2" strokeLinecap="round"
                  pointerEvents="none"/>
              )}
            </g>
          );
        })}

        {/* Drawing preview (fold/cut) */}
        {drawStart && hoverPt && (
          <line x1={drawStart.x} y1={drawStart.y} x2={hoverPt.x} y2={hoverPt.y}
            stroke={tool === "fold" ? "#2563a6" : "#c0392b"} strokeWidth="2"
            strokeDasharray={tool === "fold" ? "8 4" : "0"} opacity="0.6"/>
        )}

        {/* Flap rectangle preview while dragging */}
        {flapStart && flapCurrent && (() => {
          const x1 = Math.min(flapStart.x, flapCurrent.x);
          const x2 = Math.max(flapStart.x, flapCurrent.x);
          const y1 = Math.min(flapStart.y, flapCurrent.y);
          const y2 = Math.max(flapStart.y, flapCurrent.y);
          return (
            <g pointerEvents="none">
              <rect x={x1} y={y1} width={x2-x1} height={y2-y1}
                fill="#2563a6" fillOpacity="0.08"
                stroke="#3a3528" strokeWidth="1" strokeDasharray="3 3"/>
              {/* Show the would-be hinge in blue (default: bottom) */}
              <line x1={x1} y1={y2} x2={x2} y2={y2}
                stroke="#2563a6" strokeWidth="2.5" strokeDasharray="8 4" opacity="0.7"/>
              {/* The 3 cut sides in red */}
              <line x1={x1} y1={y1} x2={x2} y2={y1} stroke="#c0392b" strokeWidth="2" opacity="0.7"/>
              <line x1={x2} y1={y1} x2={x2} y2={y2} stroke="#c0392b" strokeWidth="2" opacity="0.7"/>
              <line x1={x1} y1={y1} x2={x1} y2={y2} stroke="#c0392b" strokeWidth="2" opacity="0.7"/>
            </g>
          );
        })()}

        {/* Outline vertices (in edit mode) */}
        {tool === "edit" && outline.map((v, i) => (
          <circle key={i} cx={v.x} cy={v.y} r="6"
            fill="#f4ecd6" stroke="#3a3528" strokeWidth="2"
            style={{ cursor: "grab" }}/>
        ))}

        {/* Snap hover indicator */}
        {hoverPt && drawStart && (
          <circle cx={hoverPt.x} cy={hoverPt.y} r="5" fill="none" stroke="#3a3528" strokeWidth="1.5"/>
        )}
      </svg>
    </div>
  );
}

// ---------- 3D Viewer ----------
function Viewer3D({ graph, lineProps, lines }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const meshGroupRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const animFrameRef = useRef(null);
  const orbitState = useRef({
    azimuth: -0.6,
    elevation: 0.5,
    distance: 600,
    target: new THREE.Vector3(0, 0, 0),
    isDragging: false,
    isPanning: false,
    lastX: 0,
    lastY: 0,
  });

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const w = mount.clientWidth;
    const h = mount.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xece4cf);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, w / h, 1, 5000);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio);
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lights — hemisphere light gives directional illumination from all
    // sides so the sheet reads as solid no matter which way it's folded.
    const hemi = new THREE.HemisphereLight(0xffffff, 0xc0b890, 0.95);
    hemi.position.set(0, 1, 0);
    scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 0.6);
    dir.position.set(300, 500, 400);
    scene.add(dir);
    const fill = new THREE.DirectionalLight(0xffffff, 0.4);
    fill.position.set(-300, -200, -200);
    scene.add(fill);

    const meshGroup = new THREE.Group();
    scene.add(meshGroup);
    meshGroupRef.current = meshGroup;

    // Camera control
    function updateCamera() {
      const o = orbitState.current;
      const x = o.distance * Math.cos(o.elevation) * Math.sin(o.azimuth);
      const y = o.distance * Math.sin(o.elevation);
      const z = o.distance * Math.cos(o.elevation) * Math.cos(o.azimuth);
      camera.position.set(o.target.x + x, o.target.y + y, o.target.z + z);
      camera.lookAt(o.target);
    }
    updateCamera();

    const dom = renderer.domElement;
    function onDown(e) {
      orbitState.current.lastX = e.clientX;
      orbitState.current.lastY = e.clientY;
      if (e.button === 2 || e.shiftKey) {
        orbitState.current.isPanning = true;
      } else {
        orbitState.current.isDragging = true;
      }
    }
    function onMove(e) {
      const o = orbitState.current;
      const dx = e.clientX - o.lastX;
      const dy = e.clientY - o.lastY;
      if (o.isDragging) {
        o.azimuth -= dx * 0.01;
        o.elevation += dy * 0.01;
        o.elevation = Math.max(-Math.PI/2 + 0.05, Math.min(Math.PI/2 - 0.05, o.elevation));
        updateCamera();
      } else if (o.isPanning) {
        const right = new THREE.Vector3();
        const up = new THREE.Vector3();
        camera.getWorldDirection(new THREE.Vector3());
        right.setFromMatrixColumn(camera.matrix, 0);
        up.setFromMatrixColumn(camera.matrix, 1);
        const panSpeed = o.distance * 0.0015;
        o.target.addScaledVector(right, -dx * panSpeed);
        o.target.addScaledVector(up, dy * panSpeed);
        updateCamera();
      }
      o.lastX = e.clientX;
      o.lastY = e.clientY;
    }
    function onUp() {
      orbitState.current.isDragging = false;
      orbitState.current.isPanning = false;
    }
    function onWheel(e) {
      e.preventDefault();
      orbitState.current.distance *= e.deltaY > 0 ? 1.1 : 0.9;
      orbitState.current.distance = Math.max(100, Math.min(2500, orbitState.current.distance));
      updateCamera();
    }
    function onContext(e) { e.preventDefault(); }

    dom.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    dom.addEventListener("wheel", onWheel, { passive: false });
    dom.addEventListener("contextmenu", onContext);

    function animate() {
      animFrameRef.current = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    }
    animate();

    function onResize() {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    const ro = new ResizeObserver(onResize);
    ro.observe(mount);

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      dom.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      dom.removeEventListener("wheel", onWheel);
      dom.removeEventListener("contextmenu", onContext);
      ro.disconnect();
      mount.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  // Rebuild meshes on graph/lineProps change
  useEffect(() => {
    const meshGroup = meshGroupRef.current;
    if (!meshGroup) return;
    while (meshGroup.children.length) {
      const c = meshGroup.children.pop();
      c.geometry?.dispose();
      c.material?.dispose();
    }
    if (!graph || graph.faces.length === 0) return;

    const folded = computeFolded(graph, lineProps);
    const { faces, transforms, verts } = folded;

    // Center/scale: compute bbox of all faces in 2D for centering
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    verts.forEach(v => {
      minX = Math.min(minX, v.x); maxX = Math.max(maxX, v.x);
      minY = Math.min(minY, v.y); maxY = Math.max(maxY, v.y);
    });
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;

    const thickness = 4;

    faces.forEach((f, fi) => {
      const T = transforms[fi];
      if (!T) return;
      const poly2D = f.verts.map(vi => verts[vi]);
      const tris = triangulate(poly2D);
      if (tris.length === 0) return;

      const vertCount = poly2D.length;
      const localOf = new Map();
      f.verts.forEach((vi, k) => localOf.set(vi, k));

      // Build positions: top ring (z=0) then bottom ring (z=-thickness),
      // each transformed by T, then centered on the bbox.
      const positions = [];
      const tmp = new THREE.Vector3();
      for (let i = 0; i < vertCount; i++) {
        const p = poly2D[i];
        tmp.set(p.x, p.y, 0).applyMatrix4(T);
        positions.push(tmp.x - cx, tmp.y - cy, tmp.z);
      }
      for (let i = 0; i < vertCount; i++) {
        const p = poly2D[i];
        tmp.set(p.x, p.y, -thickness).applyMatrix4(T);
        positions.push(tmp.x - cx, tmp.y - cy, tmp.z);
      }

      // Polygons are CCW in screen-y-down space, which is CW in right-handed
      // y-up space (Three.js convention). To get outward-facing normals on
      // the +Z (top) face, we REVERSE the triangle winding here.
      const indices = [];
      // Top face: reversed winding -> normal +Z
      tris.forEach(([a, b, c]) => {
        indices.push(localOf.get(a), localOf.get(c), localOf.get(b));
      });
      // Bottom face: original winding -> normal -Z
      tris.forEach(([a, b, c]) => {
        indices.push(
          vertCount + localOf.get(a),
          vertCount + localOf.get(b),
          vertCount + localOf.get(c)
        );
      });
      // Side walls: also reversed to face outward
      for (let i = 0; i < vertCount; i++) {
        const a = i;
        const b = (i + 1) % vertCount;
        const ab2 = vertCount + a, bb2 = vertCount + b;
        // Reversed from the original triangulation:
        indices.push(a, b, ab2);
        indices.push(b, bb2, ab2);
      }

      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      geom.setIndex(indices);
      geom.computeVertexNormals();

      const mat = new THREE.MeshLambertMaterial({
        color: 0xf2e8ca,
        side: THREE.DoubleSide,
        flatShading: false,
      });
      const mesh = new THREE.Mesh(geom, mat);
      meshGroup.add(mesh);

      // Crisp edge lines
      const edgeGeo = new THREE.EdgesGeometry(geom, 25);
      const edgeMat = new THREE.LineBasicMaterial({
        color: 0x6a5d3f,
        transparent: true,
        opacity: 0.45,
      });
      const edgeLines = new THREE.LineSegments(edgeGeo, edgeMat);
      meshGroup.add(edgeLines);
    });

    // Render flap hole outlines: for each flap, find its host face and
    // draw the flap rectangle outline using the host's transform. This
    // shows where the hole is in the main sheet even after the flap folds
    // away.
    if (lines && lines.length > 0) {
      // Group lines by flapId
      const flapGroups = new Map();
      for (const ln of lines) {
        if (!ln.flapId) continue;
        if (!flapGroups.has(ln.flapId)) flapGroups.set(ln.flapId, []);
        flapGroups.get(ln.flapId).push(ln);
      }

      for (const [flapId, flapLines] of flapGroups.entries()) {
        if (flapLines.length !== 4) continue;
        const hingeLine = flapLines.find(l => l.type === "fold");
        if (!hingeLine) continue;

        // Find the graph edge whose lineId matches the hinge line
        const hingeEdgeIdx = graph.edges.findIndex(e => e.lineId === hingeLine.id);
        if (hingeEdgeIdx === -1) continue;

        // Find the two faces adjacent to that hinge edge; host is the
        // larger one (the flap is the smaller one).
        let hostFaceIdx = -1;
        for (let fi = 0; fi < graph.faceAdj.length; fi++) {
          for (const adj of graph.faceAdj[fi]) {
            if (adj.edge === hingeEdgeIdx) {
              const otherFi = adj.face;
              // pick whichever is larger
              if (graph.faces[fi].area >= graph.faces[otherFi].area) {
                hostFaceIdx = fi;
              } else {
                hostFaceIdx = otherFi;
              }
              break;
            }
          }
          if (hostFaceIdx !== -1) break;
        }
        if (hostFaceIdx === -1) continue;
        const hostT = transforms[hostFaceIdx];
        if (!hostT) continue;

        // Collect the unique corners of the flap in their natural rectangle
        // order. Each line has a flapSide (top/right/bottom/left). The
        // four corners can be reconstructed from the bounding box of all
        // line endpoints.
        const allPts = [];
        for (const ln of flapLines) { allPts.push(ln.a); allPts.push(ln.b); }
        const x1 = Math.min(...allPts.map(p => p.x));
        const x2 = Math.max(...allPts.map(p => p.x));
        const y1 = Math.min(...allPts.map(p => p.y));
        const y2 = Math.max(...allPts.map(p => p.y));
        const corners2D = [
          { x: x1, y: y1 }, // top-left
          { x: x2, y: y1 }, // top-right
          { x: x2, y: y2 }, // bottom-right
          { x: x1, y: y2 }, // bottom-left
        ];

        // Apply host transform to each corner (top + bottom of the prism)
        // and build a closed rectangle outline at z=0 (top surface) and at
        // z=-thickness (bottom surface), plus 4 vertical edges at the
        // corners — this draws the entire prism-shaped hole.
        const tmp = new THREE.Vector3();
        const topRing = corners2D.map(p => {
          tmp.set(p.x, p.y, 0).applyMatrix4(hostT);
          return new THREE.Vector3(tmp.x - cx, tmp.y - cy, tmp.z);
        });
        const botRing = corners2D.map(p => {
          tmp.set(p.x, p.y, -thickness).applyMatrix4(hostT);
          return new THREE.Vector3(tmp.x - cx, tmp.y - cy, tmp.z);
        });
        const outlinePts = [];
        // top rectangle
        for (let k = 0; k < 4; k++) {
          outlinePts.push(topRing[k], topRing[(k + 1) % 4]);
        }
        // bottom rectangle
        for (let k = 0; k < 4; k++) {
          outlinePts.push(botRing[k], botRing[(k + 1) % 4]);
        }
        // 4 verticals at corners
        for (let k = 0; k < 4; k++) {
          outlinePts.push(topRing[k], botRing[k]);
        }
        const holeGeo = new THREE.BufferGeometry().setFromPoints(outlinePts);
        const holeMat = new THREE.LineBasicMaterial({
          color: 0xc0392b, // matches the cut-line red
          transparent: true,
          opacity: 0.7,
        });
        const holeLines = new THREE.LineSegments(holeGeo, holeMat);
        meshGroup.add(holeLines);
      }
    }

    // Orient: put group so the sheet starts horizontal-ish; rotate the
    // whole group so original 2D plane becomes XZ plane (Y-up).
    meshGroup.rotation.set(-Math.PI / 2, 0, 0);
    meshGroup.position.y = 0;
  }, [graph, lineProps, lines]);

  return <div ref={mountRef} style={styles.viewer3d}/>;
}

// ============================================================
// MAIN APP
// ============================================================

const initialOutline = [
  { x: 200, y: 150 },
  { x: 600, y: 150 },
  { x: 600, y: 450 },
  { x: 200, y: 450 },
];

export default function App() {
  const [outline, setOutline] = useState(initialOutline);
  const [lines, setLines] = useState([]);
  const [lineProps, setLineProps] = useState({}); // id -> {angle, direction}
  const [selectedLineId, setSelectedLineId] = useState(null);
  const [selectedFlapId, setSelectedFlapId] = useState(null);
  const [tool, setTool] = useState("fold"); // fold | cut | edit | flap
  const [view, setView] = useState("2d"); // 2d | 3d | split

  const graph = useMemo(() => {
    if (outline.length < 3) return null;
    try {
      return buildPlanarGraph(outline, lines);
    } catch (e) {
      console.error(e);
      return null;
    }
  }, [outline, lines]);

  const selectedLine = lines.find(l => l.id === selectedLineId);
  const selectedProps = selectedLine && selectedLine.type === "fold" ? lineProps[selectedLine.id] : null;

  // Flap selection: gather all lines belonging to the flap
  const flapLines = selectedFlapId ? lines.filter(l => l.flapId === selectedFlapId) : [];
  const flapHingeLine = flapLines.find(l => l.type === "fold");
  const flapHingeProps = flapHingeLine ? lineProps[flapHingeLine.id] : null;

  function updateSelectedProp(patch) {
    if (!selectedLineId) return;
    setLineProps({
      ...lineProps,
      [selectedLineId]: { ...lineProps[selectedLineId], ...patch }
    });
  }

  function updateFlapHingeProp(patch) {
    if (!flapHingeLine) return;
    setLineProps({
      ...lineProps,
      [flapHingeLine.id]: { ...lineProps[flapHingeLine.id], ...patch }
    });
  }

  function setFlapHingeSide(newSide) {
    if (!selectedFlapId || flapLines.length !== 4) return;
    // Reassign types: the chosen side becomes fold, others become cut
    const updated = lines.map(l => {
      if (l.flapId !== selectedFlapId) return l;
      return { ...l, type: l.flapSide === newSide ? "fold" : "cut" };
    });
    // Rewrite lineProps: keep angle/direction from old hinge, attach to new hinge id
    const oldHinge = flapLines.find(l => l.type === "fold");
    const newHinge = updated.find(l => l.flapId === selectedFlapId && l.flapSide === newSide);
    const np = { ...lineProps };
    const carry = oldHinge ? np[oldHinge.id] : null;
    if (oldHinge) delete np[oldHinge.id];
    if (newHinge) np[newHinge.id] = carry || { angle: 90, direction: "valley" };
    setLines(updated);
    setLineProps(np);
  }

  function deleteSelected() {
    if (selectedFlapId) {
      const ids = lines.filter(l => l.flapId === selectedFlapId).map(l => l.id);
      setLines(lines.filter(l => l.flapId !== selectedFlapId));
      const np = { ...lineProps };
      ids.forEach(id => delete np[id]);
      setLineProps(np);
      setSelectedFlapId(null);
      return;
    }
    if (!selectedLineId) return;
    setLines(lines.filter(l => l.id !== selectedLineId));
    const np = { ...lineProps };
    delete np[selectedLineId];
    setLineProps(np);
    setSelectedLineId(null);
  }

  function reset() {
    setOutline(initialOutline);
    setLines([]);
    setLineProps({});
    setSelectedLineId(null);
    setSelectedFlapId(null);
  }

  function loadPreset(name) {
    if (name === "box") {
      // Cross-shaped foam core box net
      const o = [
        { x: 250, y: 100 },
        { x: 450, y: 100 },
        { x: 450, y: 200 },
        { x: 600, y: 200 },
        { x: 600, y: 350 },
        { x: 450, y: 350 },
        { x: 450, y: 500 },
        { x: 250, y: 500 },
        { x: 250, y: 350 },
        { x: 100, y: 350 },
        { x: 100, y: 200 },
        { x: 250, y: 200 },
      ];
      setOutline(o);
      const newLines = [
        { id: "f1", a: { x: 250, y: 200 }, b: { x: 450, y: 200 }, type: "fold" },
        { id: "f2", a: { x: 250, y: 350 }, b: { x: 450, y: 350 }, type: "fold" },
        { id: "f3", a: { x: 250, y: 200 }, b: { x: 250, y: 350 }, type: "fold" },
        { id: "f4", a: { x: 450, y: 200 }, b: { x: 450, y: 350 }, type: "fold" },
      ];
      setLines(newLines);
      setLineProps({
        f1: { angle: 90, direction: "valley" },
        f2: { angle: 90, direction: "valley" },
        f3: { angle: 90, direction: "valley" },
        f4: { angle: 90, direction: "valley" },
      });
      setSelectedLineId(null);
      setSelectedFlapId(null);
    } else if (name === "tri") {
      setOutline([
        { x: 150, y: 200 },
        { x: 650, y: 200 },
        { x: 650, y: 400 },
        { x: 150, y: 400 },
      ]);
      setLines([
        { id: "t1", a: { x: 300, y: 200 }, b: { x: 300, y: 400 }, type: "fold" },
        { id: "t2", a: { x: 500, y: 200 }, b: { x: 500, y: 400 }, type: "fold" },
      ]);
      setLineProps({
        t1: { angle: 60, direction: "valley" },
        t2: { angle: 60, direction: "mountain" },
      });
      setSelectedLineId(null);
      setSelectedFlapId(null);
    } else if (name === "blank") {
      reset();
    } else if (name === "blankLong") {
      // Long, narrower sheet for elongated designs (display panels, signage, etc.)
      setOutline([
        { x: 80, y: 230 },
        { x: 720, y: 230 },
        { x: 720, y: 370 },
        { x: 80, y: 370 },
      ]);
      setLines([]);
      setLineProps({});
      setSelectedLineId(null);
      setSelectedFlapId(null);
    }
  }

  return (
    <div style={styles.app}>
      <style>{globalCSS}</style>

      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <div style={styles.logo}>
            <svg width="28" height="28" viewBox="0 0 28 28">
              <rect x="3" y="3" width="22" height="22" fill="none" stroke="#3a3528" strokeWidth="1.5"/>
              <line x1="3" y1="14" x2="25" y2="14" stroke="#2563a6" strokeWidth="1.5" strokeDasharray="2 2"/>
              <line x1="14" y1="3" x2="14" y2="25" stroke="#c0392b" strokeWidth="1.5"/>
            </svg>
          </div>
          <div>
            <h1 style={styles.title}>FOAMCORE</h1>
            <div style={styles.subtitle}>mockup simulator · v0.1</div>
          </div>
        </div>
        <div style={styles.viewToggle}>
          <button onClick={() => setView("2d")} style={{...styles.viewBtn, ...(view==="2d"?styles.viewBtnActive:{})}}>2D PLAN</button>
          <button onClick={() => setView("split")} style={{...styles.viewBtn, ...(view==="split"?styles.viewBtnActive:{})}}>SPLIT</button>
          <button onClick={() => setView("3d")} style={{...styles.viewBtn, ...(view==="3d"?styles.viewBtnActive:{})}}>3D PREVIEW</button>
        </div>
      </header>

      <div style={styles.main}>
        {/* Left tools */}
        <aside style={styles.sidebar}>
          <SidebarSection title="TOOL">
            <ToolButton active={tool==="edit"} onClick={() => setTool("edit")} label="EDIT OUTLINE" hint="Drag corners" />
            <ToolButton active={tool==="fold"} onClick={() => setTool("fold")} label="FOLD LINE" color="#2563a6" hint="Click & drag" />
            <ToolButton active={tool==="cut"} onClick={() => setTool("cut")} label="CUT LINE" color="#c0392b" hint="Click & drag" />
            <ToolButton active={tool==="flap"} onClick={() => setTool("flap")} label="INTERNAL FLAP" hint="Drag a rectangle" />
          </SidebarSection>

          <SidebarSection title={selectedFlapId ? "FLAP PROPERTIES" : "FOLD PROPERTIES"}>
            {selectedFlapId && flapHingeProps ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <div>
                  <label style={styles.label}>HINGE SIDE</label>
                  <div style={styles.hingeGrid}>
                    {["top", "bottom", "left", "right"].map(side => {
                      const isOn = flapHingeLine?.flapSide === side;
                      return (
                        <button key={side}
                          onClick={() => setFlapHingeSide(side)}
                          style={{...styles.hingeBtn, ...(isOn?styles.hingeBtnActive:{})}}>
                          {side.toUpperCase()}
                        </button>
                      );
                    })}
                  </div>
                </div>
                <div>
                  <label style={styles.label}>DIRECTION</label>
                  <div style={styles.segmented}>
                    <button
                      onClick={() => updateFlapHingeProp({ direction: "valley" })}
                      style={{...styles.segBtn, ...(flapHingeProps.direction==="valley"?styles.segBtnActive:{})}}>
                      ▽ INWARD
                    </button>
                    <button
                      onClick={() => updateFlapHingeProp({ direction: "mountain" })}
                      style={{...styles.segBtn, ...(flapHingeProps.direction==="mountain"?styles.segBtnActive:{})}}>
                      △ OUTWARD
                    </button>
                  </div>
                </div>
                <div>
                  <label style={styles.label}>ANGLE: {flapHingeProps.angle}°</label>
                  <input type="range" min="0" max="180" value={flapHingeProps.angle}
                    onChange={(e) => updateFlapHingeProp({ angle: parseInt(e.target.value) })}
                    style={styles.slider}/>
                  <div style={styles.angleMarks}>
                    <span>0°</span><span>90°</span><span>180°</span>
                  </div>
                  <input type="number" min="0" max="180" value={flapHingeProps.angle}
                    onChange={(e) => updateFlapHingeProp({ angle: Math.max(0, Math.min(180, parseInt(e.target.value)||0)) })}
                    style={styles.numInput}/>
                </div>
                <button onClick={deleteSelected} style={styles.deleteBtn}>✕ DELETE FLAP</button>
              </div>
            ) : selectedProps ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <div>
                  <label style={styles.label}>DIRECTION</label>
                  <div style={styles.segmented}>
                    <button
                      onClick={() => updateSelectedProp({ direction: "valley" })}
                      style={{...styles.segBtn, ...(selectedProps.direction==="valley"?styles.segBtnActive:{})}}>
                      ▽ VALLEY
                    </button>
                    <button
                      onClick={() => updateSelectedProp({ direction: "mountain" })}
                      style={{...styles.segBtn, ...(selectedProps.direction==="mountain"?styles.segBtnActive:{})}}>
                      △ MOUNTAIN
                    </button>
                  </div>
                </div>
                <div>
                  <label style={styles.label}>ANGLE: {selectedProps.angle}°</label>
                  <input type="range" min="0" max="180" value={selectedProps.angle}
                    onChange={(e) => updateSelectedProp({ angle: parseInt(e.target.value) })}
                    style={styles.slider}/>
                  <div style={styles.angleMarks}>
                    <span>0°</span><span>90°</span><span>180°</span>
                  </div>
                  <input type="number" min="0" max="180" value={selectedProps.angle}
                    onChange={(e) => updateSelectedProp({ angle: Math.max(0, Math.min(180, parseInt(e.target.value)||0)) })}
                    style={styles.numInput}/>
                </div>
                <button onClick={deleteSelected} style={styles.deleteBtn}>✕ DELETE LINE</button>
              </div>
            ) : selectedLine && selectedLine.type === "cut" ? (
              <div>
                <div style={styles.smallNote}>Cut line selected. Cuts separate the sheet — no parameters.</div>
                <button onClick={deleteSelected} style={{...styles.deleteBtn, marginTop: 12}}>✕ DELETE LINE</button>
              </div>
            ) : (
              <div style={styles.emptyHint}>Select a fold line or flap<br/>to edit its parameters</div>
            )}
          </SidebarSection>

          <SidebarSection title="PRESETS">
            <button onClick={() => loadPreset("blank")} style={styles.presetBtn}>Blank Sheet</button>
            <button onClick={() => loadPreset("blankLong")} style={styles.presetBtn}>Long Sheet</button>
            <button onClick={() => loadPreset("box")} style={styles.presetBtn}>Box Net (cross)</button>
            <button onClick={() => loadPreset("tri")} style={styles.presetBtn}>Tri-Fold Panel</button>
          </SidebarSection>
        </aside>

        {/* Canvas area */}
        <section style={styles.canvasArea}>
          {view === "2d" && (
            <Editor2D
              outline={outline} setOutline={setOutline}
              lines={lines} setLines={setLines}
              lineProps={lineProps} setLineProps={setLineProps}
              selectedLineId={selectedLineId} setSelectedLineId={setSelectedLineId}
              selectedFlapId={selectedFlapId} setSelectedFlapId={setSelectedFlapId}
              tool={tool} setTool={setTool}
              deleteSelected={deleteSelected}
            />
          )}
          {view === "3d" && <Viewer3D graph={graph} lineProps={lineProps} lines={lines}/>}
          {view === "split" && (
            <div style={styles.splitWrap}>
              <div style={styles.splitHalf}>
                <div style={styles.paneLabel}>2D PLAN</div>
                <Editor2D
                  outline={outline} setOutline={setOutline}
                  lines={lines} setLines={setLines}
                  lineProps={lineProps} setLineProps={setLineProps}
                  selectedLineId={selectedLineId} setSelectedLineId={setSelectedLineId}
                  selectedFlapId={selectedFlapId} setSelectedFlapId={setSelectedFlapId}
                  tool={tool} setTool={setTool}
                  deleteSelected={deleteSelected}
                />
              </div>
              <div style={styles.splitHalf}>
                <div style={styles.paneLabel}>3D PREVIEW</div>
                <Viewer3D graph={graph} lineProps={lineProps} lines={lines}/>
              </div>
            </div>
          )}

          {/* Bottom legend / hints */}
          <div style={styles.bottomBar}>
            <div style={styles.legendItem}>
              <svg width="32" height="8"><line x1="2" y1="4" x2="30" y2="4" stroke="#2563a6" strokeWidth="2.5" strokeDasharray="6 3"/></svg>
              <span>FOLD</span>
            </div>
            <div style={styles.legendItem}>
              <svg width="32" height="8"><line x1="2" y1="4" x2="30" y2="4" stroke="#c0392b" strokeWidth="2.5"/></svg>
              <span>CUT</span>
            </div>
            <div style={styles.legendDivider}/>
            <div style={styles.hintText}>
              {view !== "3d" && tool === "fold" && "Drag from edge to edge — click an existing line to select it · hold Shift to draw over."}
              {view !== "3d" && tool === "cut" && "Cuts separate pieces. Drag edge to edge · click a line to select · Shift to draw over."}
              {view !== "3d" && tool === "edit" && "Drag the white outline corners to reshape the sheet."}
              {view !== "3d" && tool === "flap" && "Drag a rectangle inside the sheet — 3 sides become cuts, 1 becomes the hinge fold."}
              {view === "3d" && "Drag to orbit · Shift+drag to pan · Scroll to zoom"}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

function SidebarSection({ title, children }) {
  return (
    <div style={styles.section}>
      <div style={styles.sectionTitle}>{title}</div>
      <div style={styles.sectionBody}>{children}</div>
    </div>
  );
}

function ToolButton({ active, onClick, label, color, hint }) {
  return (
    <button onClick={onClick} style={{
      ...styles.toolBtn,
      ...(active ? styles.toolBtnActive : {}),
      ...(color && active ? { borderLeftColor: color } : {})
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        {color && <span style={{ width: 12, height: 12, background: color, borderRadius: 2, display: "inline-block" }}/>}
        <span>{label}</span>
      </div>
      {hint && <div style={styles.toolHint}>{hint}</div>}
    </button>
  );
}

function Stat({ label, value }) {
  return (
    <div style={styles.statRow}>
      <span style={styles.statLabel}>{label}</span>
      <span style={styles.statValue}>{value}</span>
    </div>
  );
}

// ============================================================
// STYLES
// ============================================================

const globalCSS = `
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Bodoni+Moda:ital,wght@0,400;0,700;1,400&display=swap');

  * { box-sizing: border-box; }
  body { margin: 0; }
  button { font-family: inherit; cursor: pointer; }
  input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 4px;
    background: #d4c9a8;
    border-radius: 2px;
    outline: none;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #3a3528;
    cursor: pointer;
    border: 2px solid #ece4cf;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
  }
  input[type="range"]::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #3a3528;
    cursor: pointer;
    border: 2px solid #ece4cf;
  }
`;

const styles = {
  app: {
    width: "100%",
    height: "100vh",
    minHeight: 700,
    background: "#ece4cf",
    color: "#3a3528",
    fontFamily: "'JetBrains Mono', monospace",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "14px 22px",
    borderBottom: "1.5px solid #3a3528",
    background: "#e3d9bd",
  },
  headerLeft: {
    display: "flex",
    alignItems: "center",
    gap: 14,
  },
  logo: {
    width: 36, height: 36,
    display: "flex", alignItems: "center", justifyContent: "center",
    background: "#f4ecd6",
    border: "1.5px solid #3a3528",
  },
  title: {
    fontFamily: "'Bodoni Moda', serif",
    fontSize: 22,
    fontWeight: 700,
    margin: 0,
    letterSpacing: "0.18em",
  },
  subtitle: {
    fontSize: 9,
    letterSpacing: "0.25em",
    opacity: 0.6,
    fontStyle: "italic",
    fontFamily: "'Bodoni Moda', serif",
    marginTop: 2,
  },
  viewToggle: {
    display: "flex",
    gap: 0,
    border: "1.5px solid #3a3528",
    background: "#f4ecd6",
  },
  viewBtn: {
    padding: "7px 16px",
    background: "transparent",
    border: "none",
    borderRight: "1.5px solid #3a3528",
    fontSize: 10,
    letterSpacing: "0.15em",
    fontWeight: 500,
    color: "#3a3528",
  },
  viewBtnActive: {
    background: "#3a3528",
    color: "#f4ecd6",
  },
  main: {
    flex: 1,
    display: "flex",
    minHeight: 0,
  },
  sidebar: {
    width: 260,
    minWidth: 260,
    borderRight: "1.5px solid #3a3528",
    background: "#e3d9bd",
    overflowY: "auto",
    padding: "0",
  },
  section: {
    borderBottom: "1.5px solid #3a3528",
  },
  sectionTitle: {
    padding: "10px 16px 8px",
    fontSize: 10,
    letterSpacing: "0.22em",
    fontWeight: 700,
    background: "#d8ceb0",
    borderBottom: "1px dashed #b8a880",
  },
  sectionBody: {
    padding: 12,
    display: "flex",
    flexDirection: "column",
    gap: 8,
  },
  toolBtn: {
    background: "#f4ecd6",
    border: "1.5px solid #3a3528",
    borderLeft: "4px solid #f4ecd6",
    padding: "10px 12px",
    fontSize: 11,
    letterSpacing: "0.12em",
    textAlign: "left",
    color: "#3a3528",
    fontWeight: 500,
    transition: "all 0.15s",
  },
  toolBtnActive: {
    background: "#3a3528",
    color: "#f4ecd6",
    borderLeft: "4px solid #c0a060",
  },
  toolHint: {
    fontSize: 9,
    opacity: 0.6,
    fontStyle: "italic",
    fontFamily: "'Bodoni Moda', serif",
    marginTop: 4,
    letterSpacing: "0.05em",
  },
  label: {
    display: "block",
    fontSize: 9,
    letterSpacing: "0.18em",
    fontWeight: 700,
    marginBottom: 6,
    opacity: 0.75,
  },
  segmented: {
    display: "flex",
    border: "1.5px solid #3a3528",
  },
  segBtn: {
    flex: 1,
    padding: "8px 4px",
    background: "#f4ecd6",
    border: "none",
    borderRight: "1.5px solid #3a3528",
    fontSize: 10,
    letterSpacing: "0.1em",
    fontWeight: 500,
    color: "#3a3528",
  },
  segBtnActive: {
    background: "#3a3528",
    color: "#f4ecd6",
  },
  hingeGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 4,
  },
  hingeBtn: {
    padding: "8px 6px",
    background: "#f4ecd6",
    border: "1.5px solid #3a3528",
    fontSize: 10,
    letterSpacing: "0.1em",
    fontWeight: 500,
    color: "#3a3528",
  },
  hingeBtnActive: {
    background: "#2563a6",
    color: "#f4ecd6",
    borderColor: "#2563a6",
  },
  slider: { width: "100%", margin: "4px 0" },
  angleMarks: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: 8,
    opacity: 0.5,
    marginTop: 2,
    letterSpacing: "0.1em",
  },
  numInput: {
    width: "100%",
    marginTop: 8,
    padding: "6px 8px",
    background: "#f4ecd6",
    border: "1.5px solid #3a3528",
    fontSize: 12,
    fontFamily: "'JetBrains Mono', monospace",
    textAlign: "center",
  },
  deleteBtn: {
    padding: "8px",
    background: "transparent",
    border: "1.5px dashed #c0392b",
    color: "#c0392b",
    fontSize: 10,
    letterSpacing: "0.15em",
    fontWeight: 500,
  },
  emptyHint: {
    fontSize: 10,
    fontStyle: "italic",
    fontFamily: "'Bodoni Moda', serif",
    opacity: 0.55,
    textAlign: "center",
    padding: "8px 0",
    letterSpacing: "0.05em",
    lineHeight: 1.6,
  },
  smallNote: {
    fontSize: 10,
    fontStyle: "italic",
    fontFamily: "'Bodoni Moda', serif",
    opacity: 0.7,
    lineHeight: 1.5,
  },
  presetBtn: {
    padding: "8px 12px",
    background: "#f4ecd6",
    border: "1.5px solid #3a3528",
    fontSize: 11,
    textAlign: "left",
    color: "#3a3528",
    fontFamily: "'Bodoni Moda', serif",
    fontStyle: "italic",
    letterSpacing: "0.05em",
  },
  statRow: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: 10,
    padding: "3px 0",
  },
  statLabel: { opacity: 0.6, letterSpacing: "0.1em" },
  statValue: { fontWeight: 700 },
  canvasArea: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    minWidth: 0,
    background: "#ece4cf",
  },
  canvasWrap: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: 0,
    background: "#ece4cf",
    overflow: "hidden",
  },
  canvas: {
    width: "100%",
    height: "100%",
    maxWidth: "100%",
    background: "#ece4cf",
    display: "block",
  },
  viewer3d: {
    flex: 1,
    width: "100%",
    height: "100%",
    minHeight: 400,
  },
  splitWrap: {
    flex: 1,
    display: "flex",
    minHeight: 0,
  },
  splitHalf: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    borderRight: "1.5px solid #3a3528",
    minWidth: 0,
    position: "relative",
  },
  paneLabel: {
    position: "absolute",
    top: 10, left: 12,
    fontSize: 9,
    letterSpacing: "0.25em",
    fontWeight: 700,
    background: "#3a3528",
    color: "#f4ecd6",
    padding: "4px 10px",
    zIndex: 10,
  },
  bottomBar: {
    display: "flex",
    alignItems: "center",
    gap: 16,
    padding: "10px 18px",
    borderTop: "1.5px solid #3a3528",
    background: "#e3d9bd",
    fontSize: 10,
    letterSpacing: "0.1em",
  },
  legendItem: {
    display: "flex", alignItems: "center", gap: 6,
    fontWeight: 700,
  },
  legendDivider: {
    width: 1, height: 16, background: "#3a3528", opacity: 0.3,
  },
  hintText: {
    fontStyle: "italic",
    fontFamily: "'Bodoni Moda', serif",
    opacity: 0.7,
    letterSpacing: "0.03em",
    fontSize: 11,
  },
};
