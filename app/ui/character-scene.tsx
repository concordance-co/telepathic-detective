"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";

type CharacterSceneProps = {
  activeSuspect: "ilya" | "mira";
  mode: "room" | "interview" | "compare" | "accuse" | "result";
  pending: boolean;
  signal: "neutral" | "recalled" | "guarded";
  accused?: "ilya" | "mira" | null;
  verdict?: "correct" | "wrong" | null;
};

// Internal render height in pixels; the canvas is stretched with
// image-rendering: pixelated, which is where the whole look comes from.
const PIXEL_HEIGHT = 232;

const PALETTE = {
  void: 0x05070a,
  hullDark: 0x11151a,
  hull: 0x1a2027,
  hullLight: 0x252d36,
  floor: 0x0d1114,
  rust: 0x4a3627,
  mint: 0xa8dc32,
  amber: 0xffbd5a,
  amberDeep: 0xb35c1e,
  cyan: 0x78d9e8,
  lampWarm: 0xffe3b8
};

type Expression = {
  browAngle: number; // radians; positive = inner ends down (frown)
  browLift: number; // vertical offset
  lidClose: number; // 0 open .. 1 closed
  mouthWidth: number;
  mouthTilt: number;
  gazeX: number;
  gazeY: number;
  headPitch: number;
  headYaw: number;
  lean: number; // torso z-rotation-ish lean; positive = toward camera
};

const EXPRESSIONS: Record<"neutral" | "recalled" | "guarded" | "caught", Expression> = {
  neutral: {
    browAngle: 0.05,
    browLift: 0,
    lidClose: 0.22,
    mouthWidth: 1,
    mouthTilt: 0,
    gazeX: 0,
    gazeY: 0,
    headPitch: 0,
    headYaw: 0,
    lean: 0
  },
  // Memory surfacing: eyes drift up and away, face opens.
  recalled: {
    browAngle: -0.16,
    browLift: 0.012,
    lidClose: 0.1,
    mouthWidth: 0.92,
    mouthTilt: 0,
    gazeX: -0.35,
    gazeY: 0.5,
    headPitch: -0.06,
    headYaw: 0.1,
    lean: -0.35
  },
  // The verdict lands: head drops, eyes fall, the performance ends.
  caught: {
    browAngle: 0.12,
    browLift: -0.01,
    lidClose: 0.62,
    mouthWidth: 0.68,
    mouthTilt: 0.16,
    gazeX: 0,
    gazeY: -1,
    headPitch: 0.17,
    headYaw: -0.05,
    lean: -0.9
  },
  // Something pressing that stays unsaid: narrowed, tight, pulled back.
  guarded: {
    browAngle: 0.34,
    browLift: -0.008,
    lidClose: 0.48,
    mouthWidth: 0.62,
    mouthTilt: 0.1,
    gazeX: 0.15,
    gazeY: -0.12,
    headPitch: 0.055,
    headYaw: -0.06,
    lean: -0.8
  }
};

type CharacterSpec = {
  skin: number;
  skinShadow: number;
  hair: number;
  hairStyle: "buzz" | "bun";
  brow: number;
  iris: number;
  uniform: number;
  uniformTrim: number;
  collar: number;
  jawWidth: number;
  headWidth: number;
  stubble: boolean;
  aged: boolean;
  scar: boolean;
  lashes: boolean;
  shoulderPlate: boolean;
  harness: boolean;
};

const SPECS: Record<"ilya" | "mira", CharacterSpec> = {
  ilya: {
    skin: 0xb08a6e,
    skinShadow: 0x8d6b52,
    hair: 0x8f8d86,
    hairStyle: "buzz",
    brow: 0x6b675e,
    iris: 0x5a7a6a,
    uniform: 0x4f5240,
    uniformTrim: 0xb3703a,
    collar: 0x3a3d30,
    jawWidth: 1.12,
    headWidth: 1.0,
    stubble: true,
    aged: true,
    scar: false,
    lashes: false,
    shoulderPlate: false,
    harness: true
  },
  mira: {
    skin: 0x9c7157,
    skinShadow: 0x7c5843,
    hair: 0x241d18,
    hairStyle: "bun",
    brow: 0x241d18,
    iris: 0x4d3b2c,
    uniform: 0x2a3442,
    uniformTrim: 0x78d9e8,
    collar: 0x1d2530,
    jawWidth: 0.94,
    headWidth: 0.94,
    stubble: false,
    aged: false,
    scar: true,
    lashes: true,
    shoulderPlate: true,
    harness: false
  }
};

type CharacterRig = {
  group: THREE.Group;
  head: THREE.Group;
  torso: THREE.Group;
  face: PaintedFace;
  materials: THREE.Material[];
  nextBlink: number;
  blinkUntil: number;
  phase: number;
};

function flatMat(color: number, opts: Partial<THREE.MeshStandardMaterialParameters> = {}) {
  return new THREE.MeshStandardMaterial({
    color,
    flatShading: true,
    roughness: 0.92,
    metalness: 0.04,
    transparent: true,
    ...opts
  });
}


type FaceParams = {
  browSlope: number; // px; positive pulls the inner brow ends down
  browLift: number; // px up
  lidRows: number; // 0 open .. 6 closed
  gazeX: number; // px, negative = screen left
  gazeY: number; // px, positive = up
  mouthW: number; // half-width px
  mouthOpen: number; // px of open mouth
  mouthTilt: number; // px corner drop
  crease: boolean; // glabella lines when guarded
};

type PaintedFace = {
  texture: THREE.CanvasTexture;
  draw: (p: FaceParams) => void;
};

const FACE_W = 64;
const FACE_H = 80;

function cssColor(hex: number, alpha = 1) {
  const r = (hex >> 16) & 255;
  const g = (hex >> 8) & 255;
  const b = hex & 255;
  return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
}

function mixColor(hex: number, target: number, t: number) {
  const r = Math.round(((hex >> 16) & 255) * (1 - t) + ((target >> 16) & 255) * t);
  const g = Math.round(((hex >> 8) & 255) * (1 - t) + ((target >> 8) & 255) * t);
  const b = Math.round((hex & 255) * (1 - t) + (target & 255) * t);
  return (r << 16) | (g << 8) | b;
}

// Pixel-art face painted onto a CanvasTexture — the PS1 approach: geometry
// stays simple and the texture carries the personality. Redraws only when
// the quantized expression params actually change.
function makePaintedFace(spec: CharacterSpec): PaintedFace {
  const canvas = document.createElement("canvas");
  canvas.width = FACE_W;
  canvas.height = FACE_H;
  const ctx = canvas.getContext("2d");
  const texture = new THREE.CanvasTexture(canvas);
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.colorSpace = THREE.SRGBColorSpace;

  const skin = spec.skin;
  const shade = spec.skinShadow;
  const deep = mixColor(shade, 0x160d08, 0.55);
  const light = mixColor(skin, 0xfff2dc, 0.3);
  const mouthDark = mixColor(shade, 0x1a0c08, 0.7);
  const stubbleColor = mixColor(spec.hair, shade, 0.4);

  const px = (x: number, y: number, w: number, h: number, style: string) => {
    if (!ctx) return;
    ctx.fillStyle = style;
    ctx.fillRect(x, y, w, h);
  };

  // face silhouette half-inset per row: full through the cheeks, tapering jaw
  const insetAt = (row: number) => {
    if (row < 8) return 3 + Math.max(0, 7 - row);
    if (row <= 46) return 3;
    if (row <= 62) return 3 + Math.round((row - 46) * 0.6);
    return 13 + Math.round((row - 62) * 1.1);
  };

  let lastKey = "";

  const draw = (p: FaceParams) => {
    if (!ctx) return;
    const key = [
      p.browSlope, p.browLift, p.lidRows, p.gazeX, p.gazeY,
      p.mouthW, p.mouthOpen, p.mouthTilt, p.crease
    ].join("|");
    if (key === lastKey) return;
    lastKey = key;

    ctx.clearRect(0, 0, FACE_W, FACE_H);

    // silhouette + side shading + jawline
    for (let row = 4; row <= 75; row++) {
      const inset = insetAt(row);
      const w = FACE_W - inset * 2;
      if (w <= 6) continue;
      px(inset, row, w, 1, cssColor(skin));
      px(inset, row, 2, 1, cssColor(shade, 0.45));
      px(FACE_W - inset - 2, row, 2, 1, cssColor(shade, 0.45));
      if (row > 58) {
        px(inset, row, 1, 1, cssColor(shade, 0.8));
        px(FACE_W - inset - 1, row, 1, 1, cssColor(shade, 0.8));
      }
    }
    if (spec.aged) {
      px(15, 11, 34, 1, cssColor(shade, 0.4));
      px(18, 15, 28, 1, cssColor(shade, 0.32));
    }
    if (spec.scar) {
      px(46, 13, 1, 6, cssColor(light, 0.5));
      px(45, 15, 3, 1, cssColor(shade, 0.4));
    }
    if (p.crease) {
      px(29, 17, 1, 6, cssColor(shade, 0.55));
      px(33, 17, 1, 6, cssColor(shade, 0.55));
    }

    // brows: stepped pixel segments; positive slope knits the inner ends down
    const browY = 18 - p.browLift;
    for (let i = 0; i < 6; i++) {
      const t = (i + 0.5) / 6;
      const dyL = Math.round(p.browSlope * t);
      const dyR = Math.round(p.browSlope * (1 - t));
      px(9 + i * 3, browY + dyL, 3, 3, cssColor(spec.brow));
      px(37 + i * 3, browY + dyR, 3, 3, cssColor(spec.brow));
      px(9 + i * 3, browY + dyL + 3, 3, 1, cssColor(shade, 0.35));
      px(37 + i * 3, browY + dyR + 3, 3, 1, cssColor(shade, 0.35));
    }

    // eyes
    const eyeTop = 24;
    for (const cx of [18, 46]) {
      const x0 = cx - 7;
      px(x0, eyeTop, 15, 1, cssColor(deep, 0.65));
      px(x0, eyeTop + 1, 15, 6, cssColor(0xd8d2c4));
      px(x0, eyeTop + 1, 1, 6, cssColor(shade, 0.5));
      px(x0 + 14, eyeTop + 1, 1, 6, cssColor(shade, 0.5));

      const ix = Math.max(x0 + 2, Math.min(x0 + 8, cx - 2 + p.gazeX));
      const iy = Math.max(eyeTop + 1, Math.min(eyeTop + 3, eyeTop + 2 - p.gazeY));
      px(ix, iy, 5, 4, cssColor(spec.iris));
      px(ix + 1, iy + 1, 2, 2, cssColor(0x0d0a08));
      px(ix + 3, iy, 1, 1, cssColor(0xfffbe8, 0.85));

      if (p.lidRows > 0) {
        px(x0, eyeTop + 1, 15, Math.min(p.lidRows, 6), cssColor(skin));
        px(x0, eyeTop + Math.min(p.lidRows, 6), 15, 1, cssColor(shade, 0.85));
      }
      px(x0 + 1, eyeTop + 7, 13, 1, cssColor(shade, 0.45));
      if (spec.aged) px(x0 + 2, eyeTop + 9, 11, 1, cssColor(shade, 0.28));
      if (spec.lashes) px(x0 - 1, eyeTop, 16, 2, cssColor(0x191412, 0.9));
    }

    // nose: bridge shadow, tip highlight, wings, nostrils
    px(30, 31, 1, 15, cssColor(shade, 0.4));
    px(33, 33, 1, 12, cssColor(light, 0.25));
    px(27, 47, 2, 3, cssColor(shade, 0.5));
    px(35, 47, 2, 3, cssColor(shade, 0.5));
    px(30, 45, 4, 2, cssColor(light, 0.35));
    px(28, 49, 2, 1, cssColor(mouthDark, 0.8));
    px(34, 49, 2, 1, cssColor(mouthDark, 0.8));
    px(29, 51, 7, 1, cssColor(shade, 0.3));

    // mouth
    const my = 57;
    const half = Math.max(4, p.mouthW);
    if (p.mouthOpen > 1) {
      const h = Math.min(2 + p.mouthOpen, 8);
      px(32 - half + 1, my, half * 2 - 2, h, cssColor(mouthDark));
      px(32 - half + 2, my, half * 2 - 4, 1, cssColor(0xcfc4ae, 0.8));
      px(32 - half + 2, my + h - 1, half * 2 - 4, 1, cssColor(0x5b3a30));
    } else {
      for (let i = 0; i < 4; i++) {
        const t = Math.abs((i + 0.5) / 4 - 0.5) * 2;
        const dy = Math.round(p.mouthTilt * t);
        const segW = Math.ceil((half * 2) / 4);
        px(32 - half + i * segW, my + dy, segW, 2, cssColor(mouthDark, 0.9));
      }
      px(32 - half, my - 1, half * 2, 1, cssColor(shade, 0.35));
      px(32 - half + 1, my + 2, half * 2 - 2, 1, cssColor(light, 0.3));
    }
    px(28, 65, 8, 1, cssColor(shade, 0.3));

    // stubble stipple, deterministic
    if (spec.stubble) {
      const st = cssColor(stubbleColor, 0.5);
      for (let y = 52; y <= 73; y++) {
        const inset = insetAt(y) + 1;
        for (let x = inset; x < FACE_W - inset; x++) {
          if ((x * 7 + y * 13) % 5 === 0 && (x < 22 || x > 42 || y > 60)) {
            px(x, y, 1, 1, st);
          }
        }
      }
      px(5, 34, 2, 16, cssColor(stubbleColor, 0.35));
      px(57, 34, 2, 16, cssColor(stubbleColor, 0.35));
    }

    texture.needsUpdate = true;
  };

  draw({ browSlope: 0, browLift: 0, lidRows: 1, gazeX: 0, gazeY: 0, mouthW: 10, mouthOpen: 0, mouthTilt: 0, crease: false });

  return { texture, draw };
}

function buildCharacter(spec: CharacterSpec): CharacterRig {
  const group = new THREE.Group();
  const materials: THREE.Material[] = [];
  const mat = (color: number, opts: Partial<THREE.MeshStandardMaterialParameters> = {}) => {
    const material = flatMat(color, { emissive: color, emissiveIntensity: 0.5, ...opts });
    materials.push(material);
    return material;
  };

  const skinMat = mat(spec.skin);
  const jawMat = mat(spec.stubble ? spec.skinShadow : spec.skin);
  const hairMat = mat(spec.hair, { roughness: 1, metalness: 0, emissiveIntensity: 0.25 });
  const uniformMat = mat(spec.uniform);

  // ---- torso, chest-up (seated) ----
  const torso = new THREE.Group();
  const chest = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.135, 0.52, 6), uniformMat);
  chest.scale.set(1.28, 1, 0.8);
  chest.position.y = 0.22;
  torso.add(chest);

  const shoulderGeo = new THREE.SphereGeometry(0.105, 5, 4);
  const shoulderL = new THREE.Mesh(shoulderGeo, uniformMat);
  shoulderL.position.set(-0.205, 0.4, 0);
  shoulderL.scale.set(1.05, 0.85, 0.9);
  const shoulderR = shoulderL.clone();
  shoulderR.position.x = 0.205;
  torso.add(shoulderL, shoulderR);

  // upper arms falling out of frame
  const armGeo = new THREE.CylinderGeometry(0.062, 0.055, 0.34, 5);
  const armL = new THREE.Mesh(armGeo, uniformMat);
  armL.position.set(-0.245, 0.17, 0.02);
  armL.rotation.z = 0.12;
  const armR = armL.clone();
  armR.position.x = 0.245;
  armR.rotation.z = -0.12;
  torso.add(armL, armR);

  const collar = new THREE.Mesh(
    new THREE.CylinderGeometry(0.082, 0.105, 0.07, 6),
    mat(spec.collar)
  );
  collar.position.y = 0.455;
  torso.add(collar);

  if (spec.harness) {
    const strapGeo = new THREE.BoxGeometry(0.052, 0.4, 0.015);
    const strap = new THREE.Mesh(strapGeo, mat(spec.uniformTrim, { roughness: 0.7 }));
    strap.position.set(-0.1, 0.26, 0.152);
    strap.rotation.z = -0.22;
    strap.rotation.x = -0.08;
    torso.add(strap);
    const buckle = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.045, 0.02), mat(0x6e6a5e, { metalness: 0.5, roughness: 0.5 }));
    buckle.position.set(-0.135, 0.13, 0.155);
    torso.add(buckle);
  }

  if (spec.shoulderPlate) {
    const plate = new THREE.Mesh(
      new THREE.SphereGeometry(0.125, 5, 3, 0, Math.PI * 2, 0, Math.PI / 2),
      mat(0x39485c, { metalness: 0.35, roughness: 0.6 })
    );
    plate.position.set(0.215, 0.42, 0);
    plate.scale.set(1.12, 0.9, 1.0);
    torso.add(plate);
    const badge = new THREE.Mesh(
      new THREE.BoxGeometry(0.045, 0.03, 0.012),
      mat(spec.uniformTrim, { emissive: spec.uniformTrim, emissiveIntensity: 0.65 })
    );
    badge.position.set(-0.12, 0.33, 0.15);
    badge.rotation.x = -0.12;
    torso.add(badge);
  }

  group.add(torso);

  // ---- neck + head ----
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.068, 0.12, 6), skinMat);
  neck.position.y = 0.5;
  group.add(neck);

  const head = new THREE.Group();
  head.position.y = 0.615;

  // cranium
  const cranium = new THREE.Mesh(new THREE.BoxGeometry(0.19 * spec.headWidth, 0.155, 0.2), skinMat);
  cranium.position.set(0, 0.035, -0.01);
  head.add(cranium);

  // jaw: tapered 4-sided wedge
  const jaw = new THREE.Mesh(new THREE.CylinderGeometry(0.105 * spec.jawWidth, 0.052, 0.135, 4), jawMat);
  jaw.rotation.y = Math.PI / 4;
  jaw.scale.set(spec.headWidth, 1, 0.82);
  jaw.position.set(0, -0.075, 0.012);
  head.add(jaw);

  // ears
  const earGeo = new THREE.BoxGeometry(0.02, 0.05, 0.035);
  const earL = new THREE.Mesh(earGeo, skinMat);
  earL.position.set(-0.1 * spec.headWidth - 0.006, -0.005, -0.01);
  const earR = earL.clone();
  earR.position.x = 0.1 * spec.headWidth + 0.006;
  head.add(earL, earR);

  // hair
  if (spec.hairStyle === "buzz") {
    const cap = new THREE.Mesh(new THREE.BoxGeometry(0.196 * spec.headWidth, 0.055, 0.205), hairMat);
    cap.position.set(0, 0.095, -0.012);
    head.add(cap);
    const back = new THREE.Mesh(new THREE.BoxGeometry(0.196 * spec.headWidth, 0.115, 0.05), hairMat);
    back.position.set(0, 0.028, -0.095);
    head.add(back);
  } else {
    const cap = new THREE.Mesh(new THREE.BoxGeometry(0.2 * spec.headWidth, 0.075, 0.21), hairMat);
    cap.position.set(0, 0.09, -0.015);
    head.add(cap);
    const sideGeo = new THREE.BoxGeometry(0.022, 0.1, 0.19);
    const sideL = new THREE.Mesh(sideGeo, hairMat);
    sideL.position.set(-0.1 * spec.headWidth - 0.002, 0.02, -0.025);
    const sideR = sideL.clone();
    sideR.position.x = 0.1 * spec.headWidth + 0.002;
    head.add(sideL, sideR);
    const back = new THREE.Mesh(new THREE.BoxGeometry(0.2 * spec.headWidth, 0.16, 0.055), hairMat);
    back.position.set(0, 0.005, -0.1);
    head.add(back);
    const bun = new THREE.Mesh(new THREE.SphereGeometry(0.05, 5, 4), hairMat);
    bun.position.set(0, -0.01, -0.135);
    head.add(bun);
  }

  // ---- painted face ----
  const face = makePaintedFace(spec);
  const faceMat = new THREE.MeshStandardMaterial({
    map: face.texture,
    emissive: 0xffffff,
    emissiveMap: face.texture,
    emissiveIntensity: 0.32,
    transparent: true,
    roughness: 0.95,
    metalness: 0
  });
  materials.push(faceMat);
  const facePlane = new THREE.Mesh(
    new THREE.PlaneGeometry(0.19 * spec.headWidth, 0.2375),
    faceMat
  );
  facePlane.position.set(0, -0.0335, 0.098);
  head.add(facePlane);

  group.add(head);

  return {
    group,
    head,
    torso,
    face,
    materials,
    nextBlink: 1.5 + Math.random() * 3,
    blinkUntil: 0,
    phase: Math.random() * Math.PI * 2
  };
}

type Room = {
  lamp: THREE.SpotLight;
  interviewKey: THREE.PointLight;
  lampBulb: THREE.Mesh;
  warnStrips: THREE.Mesh[];
  warnLight: THREE.PointLight;
  lensLight: THREE.PointLight;
  lensCore: THREE.Mesh;
  stars: THREE.Points;
  fill: THREE.HemisphereLight;
};

function buildRoom(scene: THREE.Scene): Room {
  const disposePool: THREE.Material[] = [];
  // PS1 rooms baked their ambience into the textures; emissive self-color is
  // the cheap equivalent, so the hull never falls to pure black.
  const mat = (color: number, opts: Partial<THREE.MeshStandardMaterialParameters> = {}) => {
    const material = flatMat(color, { transparent: false, emissive: color, emissiveIntensity: 1.15, ...opts });
    disposePool.push(material);
    return material;
  };

  const hullMat = mat(PALETTE.hull);
  const hullDarkMat = mat(PALETTE.hullDark);

  // floor with grating strips
  const floor = new THREE.Mesh(new THREE.BoxGeometry(9, 0.1, 8), mat(PALETTE.floor));
  floor.position.set(0, -0.05, 0);
  scene.add(floor);
  const grateMat = mat(0x161c22, { metalness: 0.3, roughness: 0.7 });
  for (let i = -4; i <= 4; i++) {
    const strip = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.02, 8), grateMat);
    strip.position.set(i * 0.55, 0.011, 0);
    scene.add(strip);
  }

  // back wall with panels and a viewport slit
  const back = new THREE.Mesh(new THREE.BoxGeometry(9, 3.2, 0.14), hullDarkMat);
  back.position.set(0, 1.6, -1.85);
  scene.add(back);
  const panelMat = mat(PALETTE.hullLight);
  for (let i = 0; i < 6; i++) {
    const panel = new THREE.Mesh(
      new THREE.BoxGeometry(1.18, i % 2 === 0 ? 0.92 : 0.7, 0.05),
      i === 4 ? mat(0x33261c, { roughness: 1 }) : panelMat
    );
    panel.position.set(-3.1 + i * 1.28, i % 2 === 0 ? 0.95 : 1.06, -1.77);
    scene.add(panel);
  }

  // viewport: dark slab + frame + starfield behind
  const viewport = new THREE.Mesh(new THREE.BoxGeometry(4.6, 0.62, 0.06), mat(0x01040a, { roughness: 0.2, metalness: 0.1 }));
  viewport.position.set(0, 1.92, -1.84);
  scene.add(viewport);
  const frameMat = mat(0x2c343e, { metalness: 0.4, roughness: 0.6 });
  const frameTop = new THREE.Mesh(new THREE.BoxGeometry(4.78, 0.07, 0.1), frameMat);
  frameTop.position.set(0, 2.26, -1.78);
  const frameBottom = frameTop.clone();
  frameBottom.position.y = 1.58;
  scene.add(frameTop, frameBottom);

  const starGeo = new THREE.BufferGeometry();
  const starPositions = new Float32Array(95 * 3);
  for (let i = 0; i < starPositions.length; i += 3) {
    starPositions[i] = (Math.random() - 0.5) * 4.4;
    starPositions[i + 1] = 1.66 + Math.random() * 0.5;
    starPositions[i + 2] = -1.78;
  }
  starGeo.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
  const stars = new THREE.Points(
    starGeo,
    new THREE.PointsMaterial({ color: 0x9fb4b0, size: 0.016, sizeAttenuation: false })
  );
  scene.add(stars);

  // side walls
  const sideL = new THREE.Mesh(new THREE.BoxGeometry(0.14, 3.2, 8), hullMat);
  sideL.position.set(-4.2, 1.6, 0);
  const sideR = sideL.clone();
  sideR.position.x = 4.2;
  scene.add(sideL, sideR);

  // conduits on side walls
  const conduitMat = mat(0x30271d, { metalness: 0.3, roughness: 0.8 });
  for (const x of [-4.1, 4.1]) {
    for (const y of [0.5, 0.72]) {
      const pipe = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 7.4, 5), conduitMat);
      pipe.rotation.x = Math.PI / 2;
      pipe.position.set(x, y, 0);
      scene.add(pipe);
    }
  }

  // amber hazard strips (the guarded signal surface)
  const warnStrips: THREE.Mesh[] = [];
  const warnMat = flatMat(PALETTE.amberDeep, {
    transparent: false,
    emissive: PALETTE.amber,
    emissiveIntensity: 0.15
  });
  disposePool.push(warnMat);
  for (const x of [-4.08, 4.08]) {
    const strip = new THREE.Mesh(new THREE.BoxGeometry(0.05, 1.9, 0.09), warnMat);
    strip.position.set(x, 1.55, -1.2);
    scene.add(strip);
    warnStrips.push(strip);
  }

  // ceiling
  const ceiling = new THREE.Mesh(new THREE.BoxGeometry(9, 0.12, 8), hullDarkMat);
  ceiling.position.set(0, 3.05, 0);
  scene.add(ceiling);

  // interrogation table
  const tableMat = mat(0x3d4148, { metalness: 0.55, roughness: 0.45 });
  const tableTop = new THREE.Mesh(new THREE.BoxGeometry(3.7, 0.07, 1.15), tableMat);
  tableTop.position.set(0, 1.0, 1.2);
  scene.add(tableTop);
  const tableLeg = new THREE.Mesh(new THREE.BoxGeometry(3.0, 0.96, 0.12), mat(0x23262c, { metalness: 0.4, roughness: 0.7 }));
  tableLeg.position.set(0, 0.48, 1.2);
  scene.add(tableLeg);
  const tableEdge = new THREE.Mesh(new THREE.BoxGeometry(3.7, 0.035, 0.05), mat(0x565b64, { metalness: 0.6, roughness: 0.4 }));
  tableEdge.position.set(0, 1.045, 1.78);
  scene.add(tableEdge);

  // the J-Lens recorder on the table
  const lensBase = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.07, 0.045, 6), mat(0x0e1114, { metalness: 0.2, roughness: 0.85 }));
  lensBase.position.set(0.62, 1.058, 1.28);
  scene.add(lensBase);
  const lensCoreMat = flatMat(PALETTE.mint, {
    transparent: false,
    emissive: PALETTE.mint,
    emissiveIntensity: 0.8
  });
  disposePool.push(lensCoreMat);
  const lensCore = new THREE.Mesh(new THREE.OctahedronGeometry(0.035, 0), lensCoreMat);
  lensCore.position.set(0.62, 1.13, 1.28);
  scene.add(lensCore);

  // hanging cone lamp over the table
  const cordMat = mat(0x0c0f12);
  const cord = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.8, 4), cordMat);
  cord.position.set(0, 2.65, 0.9);
  scene.add(cord);
  const shade = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.3, 0.24, 7, 1, true), mat(0x14181d, { side: THREE.DoubleSide, metalness: 0.4, roughness: 0.6 }));
  shade.position.set(0, 2.22, 0.9);
  scene.add(shade);
  const bulbMat = flatMat(PALETTE.lampWarm, {
    transparent: false,
    emissive: PALETTE.lampWarm,
    emissiveIntensity: 1.4
  });
  disposePool.push(bulbMat);
  const lampBulb = new THREE.Mesh(new THREE.SphereGeometry(0.045, 6, 5), bulbMat);
  lampBulb.position.set(0, 2.15, 0.9);
  scene.add(lampBulb);

  // chairs (backs visible behind suspects)
  const chairMat = mat(0x1f242b, { metalness: 0.35, roughness: 0.7 });
  for (const x of [-1.15, 1.15]) {
    const seatBack = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.62, 0.06), chairMat);
    seatBack.position.set(x, 1.05, 0.28);
    scene.add(seatBack);
  }

  // lighting
  const fill = new THREE.HemisphereLight(0x8fa3a8, 0x141a1a, 1.5);
  scene.add(fill);

  // dim washes so the hull reads instead of vanishing
  const wallWashL = new THREE.PointLight(0x4c6a72, 3.2, 8, 1.6);
  wallWashL.position.set(-3.2, 2.3, -0.4);
  scene.add(wallWashL);
  const wallWashR = new THREE.PointLight(0x6a5a48, 2.6, 8, 1.6);
  wallWashR.position.set(3.2, 2.2, -0.2);
  scene.add(wallWashR);

  const lamp = new THREE.SpotLight(PALETTE.lampWarm, 20, 9, Math.PI / 3.6, 0.55, 1.6);
  lamp.position.set(0, 2.2, 0.9);
  const lampTarget = new THREE.Object3D();
  lampTarget.position.set(0, 0.9, 0.75);
  scene.add(lampTarget);
  lamp.target = lampTarget;
  scene.add(lamp);

  // cool rim from the viewport side
  const rim = new THREE.DirectionalLight(0x6f93a8, 1.1);
  rim.position.set(-2.5, 2.4, -2);
  scene.add(rim);

  const warnLight = new THREE.PointLight(PALETTE.amber, 0, 6, 1.8);
  warnLight.position.set(0, 1.7, 0.4);
  scene.add(warnLight);

  // soft face key from the detective's side of the table
  const faceKey = new THREE.PointLight(0xffd9b0, 7, 5, 1.7);
  faceKey.position.set(0, 1.5, 2.4);
  scene.add(faceKey);

  // close-up key that slides onto the active subject during interviews
  const interviewKey = new THREE.PointLight(0xffe0bc, 0, 4.5, 1.6);
  interviewKey.position.set(0, 1.7, 1.5);
  scene.add(interviewKey);

  const lensLight = new THREE.PointLight(PALETTE.mint, 0, 4, 1.8);
  lensLight.position.set(0.62, 1.35, 1.25);
  scene.add(lensLight);

  return { lamp, interviewKey, lampBulb, warnStrips, warnLight, lensLight, lensCore, stars, fill };
}

// The unifier the research points at: PS1 hardware dithered its 24-bit
// framebuffer down to 15-bit color. Bayer-4x4 ordered dithering + 31-level
// quantization at the low internal resolution ties every element of the
// scene into one deliberate image.
const PSXDitherShader = {
  uniforms: {
    tDiffuse: { value: null }
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    varying vec2 vUv;

    float bayer(ivec2 p) {
      int i = (p.y & 3) * 4 + (p.x & 3);
      int m[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
      return float(m[i]) / 16.0;
    }

    void main() {
      vec4 c = texture2D(tDiffuse, vUv);
      float t = bayer(ivec2(gl_FragCoord.xy)) - 0.5;
      vec3 q = floor(c.rgb * 31.0 + 0.5 + t * 0.55) / 31.0;
      gl_FragColor = vec4(clamp(q, 0.0, 1.0), c.a);
    }
  `
};

export function PolyScene({
  activeSuspect,
  mode,
  pending,
  signal,
  accused = null,
  verdict = null
}: CharacterSceneProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef({ activeSuspect, mode, pending, signal, accused, verdict });

  useEffect(() => {
    stateRef.current = { activeSuspect, mode, pending, signal, accused, verdict };
  }, [activeSuspect, mode, pending, signal, accused, verdict]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const host = canvas.parentElement;
    if (!host) return;

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    host.dataset.webgl = "initializing";
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false });
    } catch {
      host.dataset.webgl = "failed";
      return;
    }
    renderer.setPixelRatio(1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(PALETTE.void);
    scene.fog = new THREE.Fog(PALETTE.void, 4.5, 11);

    const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 40);
    camera.position.set(0, 1.62, 5.1);

    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    composer.addPass(new OutputPass());
    composer.addPass(new ShaderPass(PSXDitherShader));
    const room = buildRoom(scene);

    const rigs = {
      ilya: buildCharacter(SPECS.ilya),
      mira: buildCharacter(SPECS.mira)
    };
    rigs.ilya.group.position.set(-1.15, 0.52, 0.42);
    rigs.mira.group.position.set(1.15, 0.52, 0.42);
    rigs.ilya.group.scale.setScalar(1.45);
    rigs.mira.group.scale.setScalar(1.45);
    rigs.ilya.group.rotation.y = 0.1;
    rigs.mira.group.rotation.y = -0.1;
    scene.add(rigs.ilya.group, rigs.mira.group);

    // detective's foreground shoulder, attached to the camera
    const shoulderMat = new THREE.MeshBasicMaterial({ color: 0x05070a });
    const shoulder = new THREE.Mesh(new THREE.SphereGeometry(0.34, 6, 5), shoulderMat);
    shoulder.position.set(0.42, -0.42, -0.85);
    camera.add(shoulder);
    scene.add(camera);

    const pointer = new THREE.Vector2();
    const onPointerMove = (event: PointerEvent) => {
      pointer.x = (event.clientX / window.innerWidth - 0.5) * 2;
      pointer.y = (event.clientY / window.innerHeight - 0.5) * 2;
    };
    window.addEventListener("pointermove", onPointerMove, { passive: true });

    const resize = () => {
      const width = host.clientWidth;
      const height = host.clientHeight;
      const scale = Math.max(1, height / PIXEL_HEIGHT);
      renderer.setSize(Math.round(width / scale), Math.round(height / scale), false);
      composer.setSize(Math.round(width / scale), Math.round(height / scale));
      canvas.style.width = "100%";
      canvas.style.height = "100%";
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();

    const startedAt = performance.now();
    let frame = 0;
    let lastRenderAt = 0;
    let pendingSince = -1;
    let wasPending = false;

    const currentPose: Record<"ilya" | "mira", Expression> = {
      ilya: { ...EXPRESSIONS.neutral },
      mira: { ...EXPRESSIONS.neutral }
    };

    // Quantize with hysteresis: a face pixel only moves once the underlying
    // value has clearly crossed the boundary, so lerp noise cannot flicker it.
    const faceInts: Record<"ilya" | "mira", Record<string, number>> = { ilya: {}, mira: {} };
    const qz = (store: Record<string, number>, key: string, value: number) => {
      const prev = store[key];
      if (prev === undefined || Math.abs(value - prev) > 0.65) {
        store[key] = Math.round(value);
      }
      return store[key];
    };

    const cameraGoal = new THREE.Vector3();
    const lookGoal = new THREE.Vector3();
    const lookCurrent = new THREE.Vector3(0, 1.35, 0);

    const render = () => {
      const elapsed = (performance.now() - startedAt) / 1000;
      // Frame-rate-independent smoothing (occluded tabs tick at ~3fps).
      const dt = Math.min(Math.max((performance.now() - (lastRenderAt || performance.now())) / 1000, 0.001), 0.3);
      const damp = (rate: number) => 1 - Math.exp(-rate * dt);
      const current = stateRef.current;
      if (process.env.NODE_ENV !== "production") {
        (window as unknown as Record<string, unknown>).__charScene = {
          cam: camera.position.toArray(),
          look: lookCurrent.toArray(),
          mode: current.mode,
          active: current.activeSuspect,
          ilya: [rigs.ilya.group.position.x, (rigs.ilya.materials[0] as THREE.MeshStandardMaterial).opacity],
          mira: [rigs.mira.group.position.x, (rigs.mira.materials[0] as THREE.MeshStandardMaterial).opacity]
        };
      }
      const isInterview = current.mode === "interview";
      const isCompare = current.mode === "compare";
      const isRoom = current.mode === "room";
      const activeX = current.activeSuspect === "ilya" ? -1.15 : 1.15;
      // narrow windows need a wider two-shot or the left suspect hides
      // behind the DOM brief panel
      const wideComp = Math.min(1.5, Math.max(1, 1 + (1.82 / camera.aspect - 1) * 4));

      if (current.pending && !wasPending) pendingSince = elapsed;
      wasPending = current.pending;
      const talking = current.pending && pendingSince >= 0 && elapsed - pendingSince > 1.4;
      const thinking = current.pending && !talking;

      // ---- camera framing per mode ----
      if (isInterview) {
        // The chat console owns the center of the screen; frame the subject
        // into the open left third, three-quarter view across the table.
        cameraGoal.set(activeX + 0.45, 1.5, 2.95);
        lookGoal.set(activeX + 1.05, 1.42, 0.45);
      } else if (isCompare) {
        cameraGoal.set(0, 1.42, 3.6 * wideComp);
        lookGoal.set(0, 1.3, 0.3);
      } else if (isRoom) {
        cameraGoal.set(0, 1.45, 4.0 * wideComp);
        lookGoal.set(0, 1.22, 0.1);
      } else if (current.mode === "result" && current.accused) {
        // Verdict copy owns the left of the screen; frame the accused into
        // the open right side and dolly in slowly.
        const accusedX = current.accused === "ilya" ? -1.15 : 1.15;
        cameraGoal.set(accusedX - 0.4, 1.46, 2.3);
        lookGoal.set(accusedX - 0.92, 1.4, 0.45);
      } else {
        cameraGoal.set(0, 1.75, 4.9 * wideComp);
        lookGoal.set(0, 1.15, 0);
      }
      const sway = reducedMotion ? 0 : 1;
      cameraGoal.x += pointer.x * 0.05 * sway;
      cameraGoal.y += -pointer.y * 0.03 * sway;
      const isVerdict = current.mode === "result" && Boolean(current.accused);
      camera.position.lerp(cameraGoal, damp(isVerdict ? 0.55 : 2.6));
      lookCurrent.lerp(lookGoal, damp(isVerdict ? 0.7 : 3.2));
      camera.lookAt(lookCurrent);

      shoulder.visible = isInterview;

      // ---- character posing ----
      for (const id of ["ilya", "mira"] as const) {
        const rig = rigs[id];
        const isActive = current.activeSuspect === id;

        // in the interview only the subject in the chair is present
        const targetOpacity =
          (isInterview && !isActive) ||
          (current.mode === "result" && current.accused && current.accused !== id)
            ? 0
            : 1;
        for (const material of rig.materials) {
          const m = material as THREE.MeshStandardMaterial;
          m.opacity += (targetOpacity - m.opacity) * damp(4.5);
        }
        rig.group.visible = (rig.materials[0] as THREE.MeshStandardMaterial).opacity > 0.03;

        const target: Expression =
          current.mode === "result" && current.accused === id
            ? current.verdict === "correct"
              ? EXPRESSIONS.caught
              : EXPRESSIONS.neutral
            : isActive && !isRoom
              ? current.signal === "guarded"
                ? EXPRESSIONS.guarded
                : current.signal === "recalled"
                  ? EXPRESSIONS.recalled
                  : EXPRESSIONS.neutral
              : EXPRESSIONS.neutral;

        const pose = currentPose[id];
        const k = damp(3.2);
        pose.browAngle += (target.browAngle - pose.browAngle) * k;
        pose.browLift += (target.browLift - pose.browLift) * k;
        pose.lidClose += (target.lidClose - pose.lidClose) * k;
        pose.mouthWidth += (target.mouthWidth - pose.mouthWidth) * k;
        pose.mouthTilt += (target.mouthTilt - pose.mouthTilt) * k;
        pose.gazeX += ((isActive && thinking ? -0.5 : target.gazeX) - pose.gazeX) * k;
        pose.gazeY += ((isActive && thinking ? 0.45 : target.gazeY) - pose.gazeY) * k;
        pose.headPitch += (target.headPitch - pose.headPitch) * k;
        pose.headYaw += (target.headYaw - pose.headYaw) * k;
        pose.lean += (target.lean - pose.lean) * k;

        // idle micro-motion
        const idleYaw = reducedMotion ? 0 : Math.sin(elapsed * 0.19 + rig.phase) * 0.014;
        const idlePitch = reducedMotion ? 0 : Math.sin(elapsed * 0.14 + rig.phase * 2) * 0.005;
        const breathe = reducedMotion ? 0 : Math.sin(elapsed * 0.9 + rig.phase) * 0.0025;

        rig.head.rotation.y = pose.headYaw + idleYaw + (isActive && isInterview ? -Math.sign(activeX) * 0.06 : 0);
        rig.head.rotation.x = pose.headPitch + idlePitch;
        rig.torso.rotation.x = pose.lean * 0.05;
        rig.torso.position.y = breathe;
        rig.head.position.y = 0.615 + breathe * 1.4 + pose.lean * -0.008;
        rig.group.position.z = 0.42 + pose.lean * 0.05;

        // blink, then hand the quantized pose to the painted face
        if (elapsed > rig.nextBlink) {
          rig.blinkUntil = elapsed + 0.14;
          rig.nextBlink = elapsed + 3.4 + Math.random() * 4.2;
        }
        const lid = elapsed < rig.blinkUntil ? 1 : pose.lidClose;
        const flapWave = Math.abs(Math.sin(elapsed * 7 + rig.phase));
        const mouthOpen = isActive && talking && !reducedMotion
          ? (flapWave > 0.62 ? 5 : flapWave > 0.2 ? 3 : 0)
          : 0;
        const ints = faceInts[id];
        rig.face.draw({
          browSlope: qz(ints, "bs", pose.browAngle * 14),
          browLift: qz(ints, "bl", pose.browLift * 350),
          lidRows: Math.round(Math.min(lid, 1) * 6),
          gazeX: qz(ints, "gx", pose.gazeX * 4),
          gazeY: qz(ints, "gy", pose.gazeY * 3),
          mouthW: qz(ints, "mw", 10 * pose.mouthWidth),
          mouthOpen,
          mouthTilt: qz(ints, "mt", pose.mouthTilt * 20),
          crease: pose.browAngle > 0.18
        });
      }

      // ---- room signal lighting ----
      const flicker = reducedMotion
        ? 1
        : 1 - Math.max(0, Math.sin(elapsed * 13.7) * Math.sin(elapsed * 3.1) - 0.965) * 3;
      room.lamp.intensity =
        (isVerdict ? (current.verdict === "correct" ? 9 : 6) : 20) *
        flicker *
        (current.pending ? 1.06 : 1);
      (room.lampBulb.material as THREE.MeshStandardMaterial).emissiveIntensity = 1.4 * flicker;

      const keyTarget = isInterview
        ? 9
        : isVerdict
          ? current.verdict === "correct"
            ? 15
            : 3
          : 0;
      room.interviewKey.intensity += (keyTarget - room.interviewKey.intensity) * damp(3.2);
      const keyX = isVerdict && current.accused
        ? (current.accused === "ilya" ? -1.15 : 1.15) * 0.92
        : activeX * 0.92;
      room.interviewKey.position.x += (keyX - room.interviewKey.position.x) * damp(3.8);

      const warnTarget =
        isVerdict && current.verdict === "correct"
          ? 11
          : current.signal === "guarded"
            ? 7
            : 0;
      room.warnLight.intensity += (warnTarget - room.warnLight.intensity) * damp(3.8);
      const warnPulse =
        isVerdict && current.verdict === "correct"
          ? 0.95 + Math.sin(elapsed * 1.6) * 0.25
          : current.signal === "guarded"
            ? 0.65 + Math.sin(elapsed * 2.6) * 0.35
            : 0.15;
      for (const strip of room.warnStrips) {
        (strip.material as THREE.MeshStandardMaterial).emissiveIntensity +=
          (warnPulse - (strip.material as THREE.MeshStandardMaterial).emissiveIntensity) * damp(5);
      }

      const lensTarget = current.signal === "recalled" ? 3.6 : current.pending ? 1.6 : 0.5;
      room.lensLight.intensity += (lensTarget - room.lensLight.intensity) * damp(4.5);
      if (!reducedMotion) {
        room.lensCore.rotation.y = elapsed * 1.4;
        room.lensCore.rotation.x = elapsed * 0.9;
      }
      (room.lensCore.material as THREE.MeshStandardMaterial).emissiveIntensity = current.pending
        ? 1.2 + Math.sin(elapsed * 8) * 0.5
        : current.signal === "recalled"
          ? 1.5
          : 0.8;

      // slow star drift: the station is rotating
      if (!reducedMotion) {
        const positions = room.stars.geometry.getAttribute("position");
        for (let i = 0; i < positions.count; i++) {
          let x = positions.getX(i) + 0.00042;
          if (x > 2.3) x = -2.3;
          positions.setX(i, x);
        }
        positions.needsUpdate = true;
      }

      composer.render();
      host.dataset.webgl = "ready";
      lastRenderAt = performance.now();
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(render);
    };

    frame = window.requestAnimationFrame(render);
    // Occluded tabs stop firing rAF entirely; a low-rate watchdog keeps the
    // scene state advancing so the game never appears frozen on return.
    const watchdog = window.setInterval(() => {
      if (performance.now() - lastRenderAt > 300) render();
    }, 300);

    return () => {
      window.clearInterval(watchdog);
      window.cancelAnimationFrame(frame);
      composer.dispose();
      window.removeEventListener("pointermove", onPointerMove);
      observer.disconnect();
      delete host.dataset.webgl;
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Points) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          materials.forEach((material) => {
            const m = material as THREE.MeshStandardMaterial;
            if (m.map) m.map.dispose();
            m.dispose();
          });
        }
      });
      renderer.dispose();
    };
  }, []);

  return (
    <div className={`poly-scene poly-scene-${mode} character-scene`} aria-hidden="true">
      <canvas ref={canvasRef} />
      <div className="poly-vignette" />
      <div className="poly-scanlines" />
    </div>
  );
}
