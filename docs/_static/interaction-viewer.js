/* ODDT interaction atlas. Structures, contacts, and the renderer are served locally. */
(() => {
  "use strict";

  const root = document.querySelector(".oddt-interaction-atlas");
  if (!root) return;
  const base = new URL(root.dataset.baseUrl, document.baseURI);
  const cache = new Map();
  let libraryPromise;
  let viewer;
  let scene;
  let manifest;
  let active = 0;
  let generation = 0;
  let activated = false;
  let ready = false;
  let surfaceId;
  let currentSceneUrl;
  let mapGeneration = 0;

  root.innerHTML = `
    <div class="oddt-atlas-picker">
      <label for="oddt-interaction-type">Interaction type</label>
      <select id="oddt-interaction-type" disabled></select>
    </div>
    <div class="oddt-atlas-heading">
      <h3 class="oddt-atlas-title"></h3>
      <a class="oddt-atlas-pdb" target="_blank" rel="noopener"></a>
    </div>
    <p class="oddt-atlas-description"></p>
    <div class="oddt-atlas-tabs" role="tablist" aria-label="Interaction view">
      <button type="button" id="oddt-map-tab" role="tab" aria-selected="true"
        aria-controls="oddt-map-panel" tabindex="0">2D interaction map</button>
      <button type="button" id="oddt-structure-tab" role="tab" aria-selected="false"
        aria-controls="oddt-structure-panel" tabindex="-1">3D structure</button>
    </div>
    <div id="oddt-map-panel" role="tabpanel" aria-labelledby="oddt-map-tab">
      <div class="oddt-atlas-toolbar" aria-label="2D view controls">
        <label><input type="checkbox" data-option="protein-atoms" disabled> Show protein residue atoms</label>
      </div>
      <div class="oddt-atlas-map">Loading the 2D interaction map…</div>
      <p class="oddt-atlas-help">Protein sites surround the ligand and are labeled by residue, chain and number.
        Surface silhouettes in the simplified view are symbolic; in the atom view, R marks continuation into other residues.
        Straight dashed lines mark contacts; small gaps separate crossing contact strokes. Distances are measured in 3D.</p>
    </div>
    <div id="oddt-structure-panel" role="tabpanel" aria-labelledby="oddt-structure-tab" hidden>
    <div class="oddt-atlas-frame">
      <div class="oddt-atlas-viewport" role="img"
        aria-label="Interactive molecular structure; use the controls below to rotate and zoom."></div>
      <div class="oddt-atlas-cover">
        <p>Loading the 3D structure…</p>
        <button type="button" class="oddt-atlas-load" hidden>Retry 3D view</button>
      </div>
    </div>
    <div class="oddt-atlas-toolbar" aria-label="3D view controls">
      <button type="button" data-action="left" aria-label="Rotate left" disabled>↶</button>
      <button type="button" data-action="right" aria-label="Rotate right" disabled>↷</button>
      <button type="button" data-action="in" aria-label="Zoom in" disabled>+</button>
      <button type="button" data-action="out" aria-label="Zoom out" disabled>−</button>
      <button type="button" data-action="reset" disabled>Reset view</button>
      <label><input type="checkbox" data-option="labels" checked> Labels</label>
      <label><input type="checkbox" data-option="surface"> Pocket surface</label>
    </div>
    <p class="oddt-atlas-help">Drag to rotate · scroll or pinch to zoom · select a contact to focus it.</p>
    </div>
    <p class="oddt-atlas-status" role="status" aria-live="polite">Loading example details…</p>
    <div class="oddt-atlas-contacts" role="group" aria-label="Verified contacts"></div>
    <p class="oddt-atlas-downloads"></p>
  `;

  const find = (selector) => root.querySelector(selector);
  const picker = find("select");
  const status = find(".oddt-atlas-status");
  const viewport = find(".oddt-atlas-viewport");
  const cover = find(".oddt-atlas-cover");
  const load = find(".oddt-atlas-load");
  const labels = find('[data-option="labels"]');
  const surface = find('[data-option="surface"]');
  const contacts = find(".oddt-atlas-contacts");
  const map = find(".oddt-atlas-map");
  const proteinAtoms = find('[data-option="protein-atoms"]');
  const tabs = [...root.querySelectorAll('[role="tab"]')];

  function showTab(tab) {
    for (const item of tabs) {
      const selected = item === tab;
      item.setAttribute("aria-selected", String(selected));
      item.tabIndex = selected ? 0 : -1;
      find(`#${item.getAttribute("aria-controls")}`).hidden = !selected;
    }
    if (tab === tabs[1] && !activated) {
      activated = true;
      // An in-flight example will pick up activation after its data loads.
      if (manifest && root.dataset.state !== "loading") showExample();
    }
    if (ready && viewport.clientWidth) { viewer.resize(); viewer.render(); }
  }
  for (const [index, tab] of tabs.entries()) {
    tab.addEventListener("click", () => showTab(tab));
    tab.addEventListener("keydown", (event) => {
      let next;
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") next = tabs[1 - index];
      else if (event.key === "Home") next = tabs[0];
      else if (event.key === "End") next = tabs[tabs.length - 1];
      if (next) { event.preventDefault(); showTab(next); next.focus(); }
    });
  }

  function highlightMap() {
    for (const contact of map.querySelectorAll("[data-contact]")) {
      contact.classList.toggle("is-active", contact.dataset.contacts.split(" ").includes(String(active)));
    }
  }

  async function showMap(sceneUrl, token) {
    const request = ++mapGeneration;
    const filename = proteinAtoms.checked ? scene.depiction_atoms_file : scene.depiction_file;
    find("[data-depiction-download]").href = new URL(filename, sceneUrl);
    map.dataset.state = "loading";
    map.replaceChildren(document.createTextNode("Loading the 2D interaction map…"));
    try {
      const svg = await get(new URL(filename, sceneUrl));
      if (token !== generation || request !== mapGeneration) return;
      const documentSvg = new DOMParser().parseFromString(svg, "image/svg+xml");
      if (documentSvg.documentElement.localName !== "svg") throw new Error("Invalid 2D map");
      map.replaceChildren(document.importNode(documentSvg.documentElement, true));
      highlightMap();
      map.dataset.state = "ready";
    } catch (error) {
      if (token === generation && request === mapGeneration) {
        map.dataset.state = "error";
        map.textContent = "The 2D map could not load. Use the structure view or downloads below.";
      }
    }
  }
  proteinAtoms.addEventListener("change", () => showMap(currentSceneUrl, generation));

  function themeColors() {
    const style = getComputedStyle(root);
    // Expand Furo's short hex values for the molecular renderer.
    const color = (name) => style.getPropertyValue(name).trim()
      .replace(/^#([\da-f])([\da-f])([\da-f])$/i, "#$1$1$2$2$3$3");
    const background = color("--color-background-primary");
    const rgb = [1, 3, 5].map((offset) => parseInt(background.slice(offset, offset + 2), 16));
    const dark = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] < 128;
    return {
      background,
      foreground: color("--color-foreground-primary"),
      pocket: color("--color-foreground-primary"),
      surface: dark ? "#ffffff" : "#000000",
      surfaceOpacity: dark ? 0.45 : 0.32,
      border: color("--color-background-border"),
    };
  }

  async function get(url, json = false) {
    const key = String(url);
    if (!cache.has(key)) {
      // Scene and SVG URLs stay stable when the docs are rebuilt.
      cache.set(key, fetch(url, { cache: "no-cache" }).then((response) => {
        if (!response.ok) throw new Error(`Unable to load ${url}: ${response.status}`);
        return json ? response.json() : response.text();
      }).catch((error) => { cache.delete(key); throw error; }));
    }
    return cache.get(key);
  }

  function renderer() {
    if (!libraryPromise) {
      libraryPromise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = new URL(root.dataset.libraryUrl, document.baseURI);
        script.onload = () => {
          if (window.$3Dmol) resolve(window.$3Dmol);
          else { libraryPromise = null; script.remove(); reject(new Error("3D renderer unavailable")); }
        };
        script.onerror = () => {
          libraryPromise = null;
          script.remove();
          reject(new Error("Could not load the local 3D renderer"));
        };
        document.head.append(script);
      });
    }
    return libraryPromise;
  }

  const point = (xyz) => ({ x: xyz[0], y: xyz[1], z: xyz[2] });
  const selection = (end) => ({ model: end.role === "protein" ? 0 : 1, index: end.indices });
  const residueSelection = (end) => ({
    model: 0, chain: end.residue.chain, resi: end.residue.number,
    ...(end.residue.insertion ? { icode: end.residue.insertion } : {}),
  });
  const surfaceStyle = () => {
    const theme = themeColors();
    return { opacity: theme.surfaceOpacity, color: theme.surface };
  };
  function focusSelection(all = false) {
    const selected = all ? scene.contacts : [scene.contacts[active]];
    return { or: [{ model: 1 }, ...selected.map((contact) => selection(contact.start))] };
  }
  function resetView() {
    viewer.zoomTo(focusSelection(true));
    viewer.zoom(0.8);
    viewer.render();
  }

  function dashedContact(start, end, color, radius) {
    const length = Math.hypot(...start.map((value, i) => value - end[i]));
    const pieces = Math.max(2, Math.ceil(length / 0.3));
    const interpolate = (t) => point(start.map((value, i) => value + (end[i] - value) * t));
    for (let i = 0; i < pieces; i += 1) {
      viewer.addCylinder({ start: interpolate(i / pieces), end: interpolate((i + 0.56) / pieces),
        radius, color, fromCap: 1, toCap: 1 });
    }
  }

  function annotations() {
    if (!ready) return;
    const theme = themeColors();
    viewer.setBackgroundColor(theme.background);
    viewer.removeAllShapes();
    viewer.removeAllLabels();
    // Thin bond cylinders stay visible on devices that limit WebGL lines to 1px.
    viewer.setStyle({ model: 0 }, { stick: { radius: 0.05 } });
    // Recolor carbon only, preserving the usual element colors for heteroatoms.
    viewer.setStyle({ model: 0, elem: "C" }, { stick: { radius: 0.05, color: theme.pocket } });
    if (surfaceId !== undefined) viewer.setSurfaceMaterialStyle(surfaceId, surfaceStyle());
    viewer.setStyle({ model: 1 }, {
      stick: { radius: 0.18, colorscheme: "greenCarbon" },
      sphere: { scale: 0.24, colorscheme: "greenCarbon" },
    });
    // Single-atom ligands, such as sodium, need a visible ion sphere.
    if (scene.ligand_atom_names.length === 1) {
      viewer.setStyle({ model: 1 }, { sphere: { radius: 0.55, color: scene.color } });
    }
    for (const [index, contact] of scene.contacts.entries()) {
      const selected = index === active;
      viewer.setStyle(residueSelection(contact.start), {
        stick: { radius: 0.1, colorscheme: "grayCarbon" },
      });
      dashedContact(contact.start.position, contact.end.position, scene.color, selected ? 0.065 : 0.028);
      for (const end of [contact.start, contact.end]) {
        if (end.kind === "ring") {
          viewer.addSphere({ center: point(end.position), radius: selected ? 0.13 : 0.08, color: scene.color });
        }
      }
      if (selected && labels.checked) {
        const style = { fontSize: 12, fontColor: theme.foreground, backgroundColor: theme.background,
          backgroundOpacity: 0.88, borderThickness: 0.3, borderColor: theme.border, inFront: true };
        const shortLabel = (end) => {
          const atom = end.kind === "ring" ? `ring ${end.ring_number}` : end.atom_names[0];
          const residue = end.residue;
          return end.role === "protein" ? `${residue.name} ${residue.chain}${residue.number} ${atom}` : `${scene.component} ${atom}`;
        };
        viewer.addLabel(shortLabel(contact.start), { ...style, position: point(contact.start.position),
          screenOffset: { x: 5, y: 18 } });
        viewer.addLabel(shortLabel(contact.end), { ...style, position: point(contact.end.position),
          screenOffset: { x: 5, y: -20 } });
        const midpoint = contact.start.position.map((value, i) => (value + contact.end.position[i]) / 2);
        viewer.addLabel(`${contact.distance.toFixed(2)} Å`, {
          ...style, position: point(midpoint), screenOffset: { x: 0, y: 28 },
        });
      }
    }
    // Highlight last: another contact can involve the same residue or fused ring.
    viewer.setStyle(residueSelection(scene.contacts[active].start), {
      stick: { radius: 0.18, colorscheme: "cyanCarbon" },
    });
    for (const [index, button] of [...contacts.querySelectorAll("button")].entries()) {
      button.setAttribute("aria-pressed", String(index === active));
    }
    viewer.render();
  }

  async function drawSurface(token) {
    if (!ready) return;
    viewer.removeAllSurfaces();
    surfaceId = undefined;
    if (surface.checked) {
      try {
        const pending = viewer.addSurface(window.$3Dmol.SurfaceType.VDW,
          surfaceStyle(), { model: 0 }, { model: 0 });
        surfaceId = pending.surfid;
        await pending;
        if (token !== generation) return;
        if (!surface.checked) {
          viewer.removeAllSurfaces();
          surfaceId = undefined;
        } else if (surfaceId !== undefined) {
          viewer.setSurfaceMaterialStyle(surfaceId, surfaceStyle());
        }
        viewer.render();
      } catch (error) {
        if (token === generation) status.textContent = "Surface unavailable; the molecular view remains interactive.";
      }
    } else viewer.render();
  }

  function details(sceneUrl) {
    find(".oddt-atlas-title").textContent = scene.complex;
    find(".oddt-atlas-description").textContent = scene.description;
    const pdbLink = find(".oddt-atlas-pdb");
    pdbLink.textContent = `PDB ${scene.pdb_id} ↗`;
    pdbLink.href = `https://www.rcsb.org/structure/${scene.pdb_id}`;
    root.style.setProperty("--interaction-color", scene.color);
    contacts.replaceChildren();
    for (const [index, contact] of scene.contacts.entries()) {
      const button = document.createElement("button");
      button.type = "button";
      button.setAttribute("aria-pressed", String(index === active));
      const names = document.createElement("span");
      names.textContent = `${contact.start.label} ↔ ${contact.end.label}`;
      const distance = document.createElement("strong");
      distance.textContent = `${contact.distance.toFixed(2)} Å${contact.geometry ? ` · ${contact.geometry}` : ""}`;
      button.append(names, distance);
      button.addEventListener("click", () => {
        active = index;
        for (const [i, item] of [...contacts.querySelectorAll("button")].entries()) {
          item.setAttribute("aria-pressed", String(i === active));
        }
        highlightMap();
        if (ready) {
          annotations();
          viewer.zoomTo(focusSelection());
          viewer.zoom(0.8);
          viewer.render();
        }
      });
      contacts.append(button);
    }
    const downloads = find(".oddt-atlas-downloads");
    downloads.replaceChildren(document.createTextNode("Download: "));
    for (const [label, filename] of [["2D SVG", scene.depiction_file], ["pocket PDB", scene.protein_file], ["ligand SDF", scene.ligand_file], ["contacts & provenance", "scene.json"]]) {
      if (downloads.childNodes.length > 1) downloads.append(document.createTextNode(" · "));
      const link = document.createElement("a");
      link.textContent = label;
      link.href = new URL(filename, sceneUrl);
      link.download = "";
      if (label === "2D SVG") link.setAttribute("data-depiction-download", "");
      downloads.append(link);
    }
  }

  async function showExample() {
    const token = ++generation;
    ready = false;
    surfaceId = undefined;
    root.dataset.state = "loading";
    root.setAttribute("aria-busy", "true");
    for (const button of root.querySelectorAll("[data-action]")) button.disabled = true;
    // Prevent the previous structure being mistaken for the newly selected example.
    if (viewer) { viewer.clear(); viewer.render(); }
    status.textContent = "Loading example…";
    load.disabled = true;
    load.hidden = true;
    cover.querySelector("p").textContent = "Loading the 3D structure…";
    proteinAtoms.disabled = true;
    cover.hidden = false;
    try {
      const entry = manifest.find((item) => item.id === picker.value);
      const sceneUrl = new URL(entry.scene, base);
      const nextScene = await get(sceneUrl, true);
      if (token !== generation) return;
      scene = nextScene;
      currentSceneUrl = sceneUrl;
      proteinAtoms.disabled = !scene.depiction_atoms_file;
      if (proteinAtoms.disabled) proteinAtoms.checked = false;
      active = 0;
      root.dataset.example = scene.id;
      details(sceneUrl);
      await showMap(sceneUrl, token);
      if (token !== generation) return;
      status.textContent = `${scene.contacts.length} displayed contacts · ODDT / RDKit · distances in Å`;
      if (!activated) {
        root.dataset.state = "idle";
        return;
      }
      const [api, protein, ligand] = await Promise.all([
        renderer(), get(new URL(scene.protein_file, sceneUrl)), get(new URL(scene.ligand_file, sceneUrl)),
      ]);
      if (token !== generation) return;
      if (!viewer) {
        // Probe separately: the renderer need not attach its canvas until render().
        // Release the probe context immediately, then keep one actual viewer.
        const probe = document.createElement("canvas");
        const context = probe.getContext("webgl") || probe.getContext("experimental-webgl");
        if (!context) throw new Error("WebGL is not available");
        context.getExtension("WEBGL_lose_context")?.loseContext();
        viewer = api.createViewer(viewport, { backgroundColor: themeColors().background, antialias: true });
      }
      viewer.clear();
      viewer.addModel(protein, "pdb", { keepH: false });
      viewer.addModel(ligand, "sdf");
      ready = true;
      cover.hidden = true;
      for (const button of root.querySelectorAll("[data-action]")) button.disabled = false;
      annotations();
      resetView();
      viewer.resize();
      root.dataset.state = "ready";
      await drawSurface(token);
    } catch (error) {
      if (token !== generation) return;
      ready = false;
      root.dataset.state = "error";
      status.textContent = "The 3D view could not load. Try again in a WebGL-capable browser, or use the structure downloads below.";
      cover.hidden = false;
      cover.querySelector("p").textContent = "The 3D view could not load.";
      load.hidden = false;
      console.warn("ODDT interaction atlas:", error);
    } finally {
      if (token === generation) {
        root.setAttribute("aria-busy", "false");
        load.disabled = false;
      }
    }
  }

  picker.addEventListener("change", showExample);
  load.addEventListener("click", () => { activated = true; showExample(); });
  labels.addEventListener("change", annotations);
  surface.addEventListener("change", () => drawSurface(generation));
  for (const button of root.querySelectorAll("[data-action]")) {
    button.addEventListener("click", () => {
      if (!ready) return;
      switch (button.dataset.action) {
        case "left": viewer.rotate(-15, "y"); break;
        case "right": viewer.rotate(15, "y"); break;
        case "in": viewer.zoom(1.2); break;
        case "out": viewer.zoom(0.8); break;
        case "reset": resetView(); break;
      }
      viewer.render();
    });
  }
  new ResizeObserver(() => {
    if (ready && viewport.clientWidth) { viewer.resize(); viewer.render(); }
  }).observe(viewport);
  new MutationObserver(annotations).observe(document.documentElement, {
    attributes: true, attributeFilter: ["data-theme"], subtree: true,
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", annotations);
  get(new URL("manifest.json", base), true).then((entries) => {
    manifest = entries;
    for (const entry of manifest) {
      const option = document.createElement("option");
      option.value = entry.id;
      option.textContent = `${entry.title} · ${entry.pdb_id}`;
      picker.append(option);
    }
    picker.disabled = false;
    showExample();
  }).catch(() => {
    status.textContent = "Example data could not load. Use the structure links in the table below.";
    load.disabled = true;
  });
})();
