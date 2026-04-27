import os, json, time, datetime, subprocess, threading, shutil, math
import xml.etree.ElementTree as ET

import streamlit as st
import pandas as pd
import folium
import folium.plugins as plugins
from streamlit_folium import st_folium
from pyproj import Transformer
import requests
import traci

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SUMO TIA Orchestrator", layout="wide", page_icon="🚦")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PROJECTS_ROOT = "projects"
MODEL_NAME    = "llama3:latest"
os.makedirs(PROJECTS_ROOT, exist_ok=True)

SCENARIOS = [
    ("present",       "Present Condition",             "scene_01_present"),
    ("5yr_no_dev",    "5-Year Future (Without Dev)",   "scene_02_5yr_no_dev"),
    ("5yr_with_dev",  "5-Year Future (With Dev)",      "scene_03_5yr_with_dev"),
    ("20yr_no_dev",   "20-Year Future (Without Dev)",  "scene_04_20yr_no_dev"),
    ("20yr_with_dev", "20-Year Future (With Dev)",     "scene_05_20yr_with_dev"),
]

TRACI_PORTS = {
    "present":       8813,
    "5yr_no_dev":    8814,
    "5yr_with_dev":  8815,
    "20yr_no_dev":   8816,
    "20yr_with_dev": 8817,
}

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA  (start once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _ensure_ollama():
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        if r.status_code == 200:
            return "already_running"
    except Exception:
        pass
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        time.sleep(3)
        return "started"
    except Exception as e:
        return f"failed: {e}"

_ensure_ollama()

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT PATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def pdir(name):        return os.path.join(PROJECTS_ROOT, name)
def pdata(name):       return os.path.join(pdir(name), "data")
def pscene(name, slug):return os.path.join(pdir(name), slug)
def pjson(name):       return os.path.join(pdir(name), "project.json")
def cfg_path(name):    return os.path.join(pdata(name), "module_a_config.json")
def fcfg_path(name):   return os.path.join(pdata(name), "module_c_final_config.json")

def default_pstate(name):
    now = datetime.datetime.now().isoformat()
    return {
        "name": name, "created": now, "last_modified": now,
        "modules_completed": {
            "1_data_ingest": False, "2_trip_gen": False,
            "3_network_compile": False, "4_ai_orchestration": False,
        },
        "scenarios_compiled":  {s[0]: False for s in SCENARIOS},
        "scenarios_simulated": {s[0]: False for s in SCENARIOS},
        "scenario_results":    {},
    }

def load_pstate(name):
    p = pjson(name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return default_pstate(name)

def save_pstate(name, state):
    state["last_modified"] = datetime.datetime.now().isoformat()
    os.makedirs(pdir(name), exist_ok=True)
    with open(pjson(name), "w") as f:
        json.dump(state, f, indent=2)

def create_project(name):
    os.makedirs(pdata(name), exist_ok=True)
    for _, _, slug in SCENARIOS:
        os.makedirs(pscene(name, slug), exist_ok=True)
    state = default_pstate(name)
    save_pstate(name, state)
    return state

def list_projects():
    if not os.path.exists(PROJECTS_ROOT):
        return []
    return sorted([
        d for d in os.listdir(PROJECTS_ROOT)
        if os.path.isdir(os.path.join(PROJECTS_ROOT, d)) and os.path.exists(pjson(d))
    ])

# ─────────────────────────────────────────────────────────────────────────────
# GEO / XML HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def convert_gps_to_cartesian(lat, lon, clat, clon):
    proj = f"+proj=ortho +lat_0={clat} +lon_0={clon} +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
    t = Transformer.from_crs("epsg:4326", proj, always_xy=True)
    return t.transform(lon, lat)

def pretty_xml(elem):
    from xml.dom import minidom
    raw = ET.tostring(elem, "utf-8")
    parsed = minidom.parseString(raw)
    return "\n".join(l for l in parsed.toprettyxml(indent="    ").split("\n") if l.strip())

# ─────────────────────────────────────────────────────────────────────────────
# TRAFFIC DATA PARSER
# ─────────────────────────────────────────────────────────────────────────────
def parse_traffic_data(file):
    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file, header=None)
        elif file.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file, header=None, engine="openpyxl")
        else:
            raise ValueError("Unsupported format.")
        hdr          = df.iloc[:3, 1:].dropna(axis=1, how="all")
        vehicle_types = hdr.iloc[0].tolist()
        pcus          = hdr.iloc[1].astype(float).tolist()
        categories    = hdr.iloc[2].tolist()
        df_v          = df.iloc[3:].dropna(how="all")
        df_v          = df_v.iloc[:, :len(vehicle_types) + 1]
        df_v.columns  = ["Movement"] + vehicle_types
        df_v.reset_index(drop=True, inplace=True)
        for col in vehicle_types:
            df_v[col] = pd.to_numeric(df_v[col], errors="coerce").fillna(0).astype(int)
        return vehicle_types, pcus, categories, df_v
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None, None, None, None

def extract_legs(df_v):
    origs, dests = set(), set()
    for mov in df_v["Movement"].dropna().tolist():
        parts = mov.split(" to ")
        if len(parts) == 2:
            origs.add(parts[0].strip()); dests.add(parts[1].strip())
    return sorted(origs.union(dests))

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def check_ollama_health():
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=10); r.raise_for_status()
        names = [m.get("name") for m in r.json().get("models", [])]
        if not any(MODEL_NAME in n for n in names):
            st.error(f"❌ Model `{MODEL_NAME}` missing. Run `ollama pull {MODEL_NAME}`.")
            return False
        return True
    except Exception as e:
        st.error(f"❌ Ollama not running. ({e})"); return False

def call_ollama(messages):
    try:
        prompt = "".join(f"[{m['role'].upper()}]: {m['content']}\n\n" for m in messages) + "[ASSISTANT]:\n"
        r = requests.post("http://127.0.0.1:11434/api/generate",
                          json={"model": MODEL_NAME, "prompt": prompt, "stream": False, "format": "json"},
                          timeout=120)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        return json.dumps({"action": "error", "message": f"API Error: {e}"})

# ─────────────────────────────────────────────────────────────────────────────
# SUMO COMPILATION
# ─────────────────────────────────────────────────────────────────────────────
def _build_routes(config, scenario_name):
    """Build routes ET root for one scenario."""
    movements  = config.get("movements", [])
    vtypes     = config.get("vehicle_types", [])
    pcus       = config.get("pcus", [])
    cats       = config.get("categories", [])
    vcfgs      = config.get("vehicle_configs", {})
    gen_trips  = config.get("generated_trips", {})
    ai_matrix  = config.get("ai_matrix", [])
    agr        = float(config.get("agr", 0.0))

    dist       = {m.get("Movement",""): float(m.get("Percentage (%)",0))/100.0 for m in ai_matrix}
    years      = 5 if "5-Year" in scenario_name else (20 if "20-Year" in scenario_name else 0)
    with_dev   = "With Dev" in scenario_name
    gf         = (1 + agr/100.0) ** years

    safe_v, r_root = [], ET.Element("routes")
    for i, v in enumerate(vtypes):
        safe      = str(v).strip().replace(" ", "_").replace("/", "_")
        safe_v.append(safe)
        pcu       = pcus[i] if i < len(pcus) else 1.0
        cat       = cats[i] if i < len(cats) else "Passenger"
        cl        = str(cat).lower()
        vcls, gsh = "passenger", "passenger"
        vc        = vcfgs.get(v, {"max_speed": 60, "turn_speed": 10})
        mspd      = vc["max_speed"] / 3.6
        tspd      = str(vc["turn_speed"])
        extra     = {}
        if "rickshaw" in cl:      vcls = "bicycle";  extra["lcSpeed"] = "3.0"
        elif "bicycle" in cl or "motorcycle" in cl: vcls = "motorcycle"; extra["latAlignment"] = "compact"
        elif "heavy" in cl or "truck" in cl:  vcls, gsh = "truck", "truck"
        elif "bus" in cl:                     vcls, gsh = "bus",   "bus"
        att = {"id": safe, "length": str(round(4.5*float(pcu),2)), "minGap": "2.5",
               "maxSpeed": str(round(mspd,2)), "vClass": vcls, "guiShape": gsh}
        att.update(extra)
        ve = ET.SubElement(r_root, "vType", attrib=att)
        ET.SubElement(ve, "param", attrib={"key": "turningSpeed", "value": tspd})

    for m in movements:
        mov = m.get("Movement", "")
        parts = mov.split(" to ")
        if len(parts) == 2:
            for idx, safe in enumerate(safe_v):
                cnt = int(round(int(m.get(vtypes[idx], 0)) * gf))
                if with_dev:
                    cnt += int(round(int(gen_trips.get(vtypes[idx], 0)) * dist.get(mov, 0.0)))
                if cnt > 0:
                    ET.SubElement(r_root, "flow", attrib={
                        "id": f"flow_{parts[0].strip()}_{parts[1].strip()}_{safe}",
                        "type": safe, "begin": "0", "end": "3600", "number": str(cnt),
                        "from": f"in_{parts[0].strip()}", "to": f"out_{parts[1].strip()}",
                        "departSpeed": "max", "departLane": "best"})
    return r_root


def compile_all_scenarios(config, project_name):
    """Compile ALL 5 scenario folders. Returns (all_ok, {skey: (ok, msg)})."""
    coords, legs, lcfg = config.get("coordinates",{}), config.get("legs",[]), config.get("leg_configs",{})
    clat = coords["Intersection Center"]["lat"]
    clon = coords["Intersection Center"]["lng"]

    # ── Topology (shared) ──
    n_root = ET.Element("nodes")
    ET.SubElement(n_root, "node", attrib={"id":"center","x":"0.0","y":"0.0","type":"traffic_light"})
    for leg in legs:
        x, y = convert_gps_to_cartesian(coords[f"{leg} Endpoint"]["lat"], coords[f"{leg} Endpoint"]["lng"], clat, clon)
        ET.SubElement(n_root, "node", attrib={"id":f"end_{leg}","x":f"{x:.2f}","y":f"{y:.2f}","type":"priority"})
    nod_xml = pretty_xml(n_root)

    e_root = ET.Element("edges")
    for leg in legs:
        lc = lcfg[leg]
        ET.SubElement(e_root, "edge", attrib={"id":f"in_{leg}","from":f"end_{leg}","to":"center","priority":"2","numLanes":str(lc["lanes"]),"speed":"35.0","width":str(lc["width"])})
        ET.SubElement(e_root, "edge", attrib={"id":f"out_{leg}","from":"center","to":f"end_{leg}","priority":"1","numLanes":str(lc["lanes"]),"speed":"35.0","width":str(lc["width"])})
    edg_xml = pretty_xml(e_root)

    # ── Run netconvert once (into first scene folder) ──
    first_slug = SCENARIOS[0][2]
    fd         = pscene(project_name, first_slug)
    os.makedirs(fd, exist_ok=True)
    nod_f = os.path.join(fd, "network.nod.xml")
    edg_f = os.path.join(fd, "network.edg.xml")
    net_f = os.path.join(fd, "network.net.xml")
    with open(nod_f, "w", encoding="utf-8") as f: f.write(nod_xml)
    with open(edg_f, "w", encoding="utf-8") as f: f.write(edg_xml)
    try:
        subprocess.run(["netconvert",
            "--node-files", nod_f, "--edge-files", edg_f, "--output-file", net_f,
            "--no-turnarounds","true","--tls.guess","true","--tls.default-type","actuated"],
            check=True, capture_output=True)
    except Exception as e:
        return False, {s[0]: (False, f"netconvert failed: {e}") for s in SCENARIOS}

    # ── Per-scenario files ──
    results   = {}
    leg_fp    = "|".join(sorted(legs))
    for skey, sname, sslug in SCENARIOS:
        sd = pscene(project_name, sslug)
        os.makedirs(sd, exist_ok=True)
        try:
            for fn in ["network.nod.xml","network.edg.xml","network.net.xml"]:
                src = os.path.join(fd, fn); dst = os.path.join(sd, fn)
                if src != dst: shutil.copy2(src, dst)

            rou = _build_routes(config, sname)
            with open(os.path.join(sd,"traffic.rou.xml"),"w",encoding="utf-8") as f:
                f.write(pretty_xml(rou))

            cfg_r = ET.Element("configuration")
            inp   = ET.SubElement(cfg_r,"input")
            ET.SubElement(inp,"net-file",    attrib={"value":"network.net.xml"})
            ET.SubElement(inp,"route-files", attrib={"value":"traffic.rou.xml"})
            t = ET.SubElement(cfg_r,"time")
            ET.SubElement(t,"begin",attrib={"value":"0"})
            ET.SubElement(t,"end",  attrib={"value":"3600"})
            with open(os.path.join(sd,"sim.sumocfg"),"w",encoding="utf-8") as f:
                f.write(pretty_xml(cfg_r))
            with open(os.path.join(sd,"sim_meta.json"),"w") as f:
                json.dump({"scenario":sname,"legs_fingerprint":leg_fp}, f)
            results[skey] = (True, "OK")
        except Exception as e:
            results[skey] = (False, str(e))

    # ── Update project state ──
    ps = load_pstate(project_name)
    ps["modules_completed"]["3_network_compile"] = True
    ps["scenarios_compiled"] = {skey: results.get(skey,(False,""))[0] for skey,_,_ in SCENARIOS}
    save_pstate(project_name, ps)
    return all(v[0] for v in results.values()), results

# ─────────────────────────────────────────────────────────────────────────────
# HCM ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def get_hcm_los(d):
    if d <= 10: return "A"
    if d <= 20: return "B"
    if d <= 35: return "C"
    if d <= 55: return "D"
    if d <= 80: return "E"
    return "F"

# ─────────────────────────────────────────────────────────────────────────────
# TRACI SIMULATION THREAD  (unique label + port per scenario)
# ─────────────────────────────────────────────────────────────────────────────
def start_traci_thread(app_state, legs_list, leg_configs, sumocfg_path, scenario_key):
    def run():
        port  = TRACI_PORTS[scenario_key]
        label = f"sim_{scenario_key}_{int(time.time())}"
        cmd   = ["sumo-gui", "-c", sumocfg_path, "--start", "true",
                 "--step-length", "1.0", "--lateral-resolution", "0.8",
                 "--remote-port", str(port)]
        try:
            subprocess.Popen(cmd)
            time.sleep(2.5)
            conn = traci.connect(port=port, label=label)
        except Exception as e:
            app_state["error"] = str(e); app_state["running"] = False; return

        ms  = app_state["legs"]
        acc = {k: [] for k in legs_list}
        try:
            while app_state["running"] and conn.simulation.getMinExpectedNumber() > 0:
                if app_state["paused"]: time.sleep(0.5); continue
                conn.simulationStep()
                app_state["step"]          = int(conn.simulation.getTime())
                app_state["total_inserted"]+= conn.simulation.getDepartedNumber()
                arr  = conn.simulation.getArrivedIDList()
                app_state["total_arrived"] += len(arr)
                active = conn.vehicle.getIDList()
                speeds = []
                for k in legs_list: ms[k]["queue"] = 0
                for vid in active:
                    spd = conn.vehicle.getSpeed(vid); speeds.append(spd)
                    p   = vid.split("_")
                    if len(p) >= 4 and p[0] == "flow" and p[1] in ms:
                        acc[p[1]].append(conn.vehicle.getAccumulatedWaitingTime(vid))
                        if spd < 0.1:
                            ms[p[1]]["queue"] += 6.0 / max(1, leg_configs[p[1]]["lanes"])
                for vid in arr:
                    p = vid.split("_")
                    if len(p) >= 4 and p[0] == "flow" and p[1] in ms:
                        ms[p[1]]["vol"] += 1
                if speeds: app_state["avg_speed"] = sum(speeds)/len(speeds)*3.6
                for k in legs_list:
                    if acc[k]:
                        da = sum(acc[k])/len(acc[k])
                        ms[k]["delay_avg"] = da; ms[k]["los"] = get_hcm_los(da)
                    if ms[k]["queue"] > ms[k]["max_q"]: ms[k]["max_q"] = ms[k]["queue"]
                    if len(acc[k]) > 500: acc[k] = acc[k][-500:]
                time.sleep(0.01)
        except Exception as e:
            app_state["error"] = f"Halted: {e}"
        finally:
            app_state["running"] = False; app_state["simulation_finished"] = True
            try: conn.close()
            except: pass
    threading.Thread(target=run, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# GEO MAP FRAGMENT
# ─────────────────────────────────────────────────────────────────────────────
@st.fragment
def render_geo_map():
    if "legs" not in st.session_state:
        st.info("⬅️ Upload your traffic data first."); return
    legs  = st.session_state.legs
    if "coordinates" not in st.session_state: st.session_state.coordinates = {}
    all_locs   = ["Intersection Center"] + [f"{l} Endpoint" for l in legs] + ["Project Development Site"]
    coords     = st.session_state.coordinates
    unassigned = [l for l in all_locs if l not in coords]
    if unassigned:
        st.info(f"👆 Click the map then assign.  Remaining: **{', '.join(unassigned)}**")
    else:
        st.success("✅ All points defined!")
    ic = coords.get("Intersection Center")
    if "map_center" not in st.session_state:
        st.session_state.map_center = [ic["lat"],ic["lng"]] if ic else [23.8103,90.4125]
        st.session_state.map_zoom   = 17 if ic else 13
    m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, tiles="OpenStreetMap")
    plugins.Fullscreen(position="topright").add_to(m)
    for name, c in coords.items():
        color = "red" if "Center" in name else ("green" if "Project" in name else "blue")
        folium.Marker([c["lat"],c["lng"]], tooltip=name,
            popup=folium.Popup(f"<b>{name}</b><br>{c['lat']:.5f},{c['lng']:.5f}", max_width=200),
            icon=folium.Icon(color=color, icon="map-marker", prefix="fa")).add_to(m)
        if "Endpoint" in name and ic:
            folium.PolyLine([[ic["lat"],ic["lng"]],[c["lat"],c["lng"]]], color="#4fc3f7", weight=2, dash_array="6").add_to(m)
    md = st_folium(m, width="100%", height=500, key="geo_map_osm", returned_objects=["last_clicked"])
    if md and md.get("last_clicked"):
        lc = md["last_clicked"]
        pt = (round(lc["lat"],7), round(lc["lng"],7))
        if pt != st.session_state.get("_map_last_pt"):
            st.session_state._map_last_pt = pt
            st.session_state._map_pending = {"lat": lc["lat"], "lng": lc["lng"]}
    pending = st.session_state.get("_map_pending")
    if pending and unassigned:
        st.markdown(f"<div style='background:#0e2233;border-left:4px solid #4fc3f7;padding:8px 14px;border-radius:6px;'>"
                    f"<span style='color:#4fc3f7;font-size:11px;font-weight:700;'>CLICKED</span>&nbsp;"
                    f"<code style='color:#fff;'>{pending['lat']:.5f}, {pending['lng']:.5f}</code></div>",
                    unsafe_allow_html=True)
        a1, a2, a3 = st.columns([3,1,1])
        with a1: pick = st.selectbox("Assign as:", unassigned, key="_map_lbl", label_visibility="collapsed")
        with a2:
            if st.button("✅ Assign", type="primary", use_container_width=True, key="_map_asgn"):
                st.session_state.coordinates[pick] = pending; st.session_state._map_pending = None
                if pick == "Intersection Center":
                    st.session_state.map_center = [pending["lat"],pending["lng"]]; st.session_state.map_zoom = 17
                st.toast(f"'{pick}' saved!", icon="📍"); st.rerun()
        with a3:
            if st.button("❌", use_container_width=True, key="_map_cncl"):
                st.session_state._map_pending = None; st.rerun()
    elif pending and not unassigned: st.session_state._map_pending = None
    if coords:
        st.markdown("---"); st.markdown("**📌 Assigned Points**")
        for name, c in list(coords.items()):
            r1,r2,r3 = st.columns([3,4,1])
            with r1: st.write(f"**{name}**")
            with r2: st.caption(f"{c['lat']:.6f}, {c['lng']:.6f}")
            with r3:
                if st.button("🗑️", key=f"del_{name}"):
                    del st.session_state.coordinates[name]; st.session_state._map_pending = None; st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION HELPER
# ─────────────────────────────────────────────────────────────────────────────
def go_to(phase): st.session_state.nav = phase

# ══════════════════════════════════════════════════════════════════════════════
# PROJECT SELECTOR  — shown whenever no active project
# ══════════════════════════════════════════════════════════════════════════════
if "active_project" not in st.session_state:
    st.session_state.active_project = None

if st.session_state.active_project is None:
    st.title("🚦 TIA Micro-Sim Orchestrator")
    st.subheader("Select or create a project to begin")
    existing = list_projects()
    tab_new, tab_open = st.tabs(["➕ New Project", "📂 Open Existing"])

    with tab_new:
        c1, c2 = st.columns([3,1])
        with c1:
            new_name = st.text_input("Project Name", placeholder="e.g. Farmgate Intersection – TIA 2026", key="new_proj_name")
        with c2:
            st.write(""); st.write("")
            if st.button("Create Project", type="primary", use_container_width=True):
                nm = new_name.strip()
                if not nm:
                    st.error("Please enter a project name.")
                elif nm in existing:
                    st.error(f"Project '{nm}' already exists.")
                else:
                    create_project(nm)
                    st.session_state.active_project = nm
                    st.session_state.nav = "📌 1. Data Ingestion & Map"
                    st.rerun()

    with tab_open:
        if not existing:
            st.info("No existing projects found. Create one above.")
        else:
            for pname in existing:
                ps   = load_pstate(pname)
                mods = sum(1 for v in ps["modules_completed"].values() if v)
                comp = sum(1 for v in ps["scenarios_compiled"].values() if v)
                sims = sum(1 for v in ps["scenarios_simulated"].values() if v)
                with st.container(border=True):
                    c1,c2,c3,c4,c5 = st.columns([3,1,1,1,1])
                    with c1:
                        st.markdown(f"**{pname}**")
                        last = ps.get("last_modified","")[:16].replace("T"," ")
                        st.caption(f"Last modified: {last}")
                    with c2: st.metric("Modules",   f"{mods}/4")
                    with c3: st.metric("Compiled",  f"{comp}/5")
                    with c4: st.metric("Simulated", f"{sims}/5")
                    with c5:
                        st.write(""); st.write("")
                        if st.button("Open →", key=f"open_{pname}", type="primary", use_container_width=True):
                            st.session_state.active_project = pname
                            st.session_state.nav = "📌 1. Data Ingestion & Map"
                            # Restore session from saved config
                            cp = cfg_path(pname)
                            if os.path.exists(cp):
                                with open(cp) as f: saved = json.load(f)
                                st.session_state.legs        = saved.get("legs", [])
                                st.session_state.coordinates = saved.get("coordinates", {})
                                st.session_state.vehicle_types = saved.get("vehicle_types",[])
                                st.session_state.pcus          = saved.get("pcus",[])
                                st.session_state.categories    = saved.get("categories",[])
                                st.session_state.df_volume     = pd.DataFrame(saved.get("movements",[]))
                            st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP  — reached only when active_project is set
# ══════════════════════════════════════════════════════════════════════════════
proj   = st.session_state.active_project
pstate = load_pstate(proj)

# ── Sidebar ──
st.sidebar.title("TIA Micro-Sim Orchestrator")
st.sidebar.markdown(f"**Project:** {proj}")
mods_done = sum(1 for v in pstate["modules_completed"].values() if v)
st.sidebar.progress(mods_done / 4, text=f"Modules: {mods_done}/4")
scenes_done = sum(1 for v in pstate["scenarios_simulated"].values() if v)
st.sidebar.progress(scenes_done / 5, text=f"Scenarios simulated: {scenes_done}/5")
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Switch Project", use_container_width=True):
    st.session_state.active_project = None
    for k in ["legs","coordinates","vehicle_types","pcus","categories","df_volume","sim_states",
               "nav","chat_history","ai_matrix","agr","generated_trips","map_center","map_zoom"]:
        st.session_state.pop(k, None)
    st.rerun()

if "nav" not in st.session_state:
    st.session_state.nav = "📌 1. Data Ingestion & Map"

nav_options = ["📌 1. Data Ingestion & Map","📈 2. Trip Generation & Forecasting",
               "⚙️ 3. Network Compiler","🤖 4. AI Orchestration","🚦 5. Live Dashboard",
               "📊 6. Comparison Report"]
choice = st.sidebar.radio("Navigation", nav_options, key="nav")

# ╔════════════════════════════════════════════════════════════╗
# ║  MODULE 1 — Data Ingestion & Map                          ║
# ╚════════════════════════════════════════════════════════════╝
if choice == "📌 1. Data Ingestion & Map":
    st.title(f"Module 1: Map & Data Ingestor  —  {proj}")
    if "coordinates" not in st.session_state: st.session_state.coordinates = {}

    col1, col2 = st.columns([1,1])
    with col1:
        st.header("1. Upload Traffic Volume Data")
        uploaded_file = st.file_uploader("Upload Data", type=["csv","xlsx"])
        if uploaded_file:
            vt, pcus, cats, df_v = parse_traffic_data(uploaded_file)
            if vt is not None:
                st.success("File parsed successfully.")
                st.session_state.legs          = extract_legs(df_v)
                st.session_state.df_volume     = df_v
                st.session_state.vehicle_types = vt
                st.session_state.pcus          = pcus
                st.session_state.categories    = cats
                st.dataframe(df_v, use_container_width=True)
    with col2:
        st.header("2. Geospatial Definition")

    render_geo_map()

    if "legs" in st.session_state:
        st.header("3. Physical Leg Configurations")
        leg_configs = {}
        cols = st.columns(len(st.session_state.legs))
        for i, leg in enumerate(st.session_state.legs):
            with cols[i]:
                st.subheader(f"{leg} Approach")
                lanes = st.number_input("Entering Lanes", 1, 8, 2, key=f"lane_{leg}")
                width = st.number_input("Lane Width (m)", 2.0, 6.0, 3.3, 0.1, key=f"width_{leg}")
                leg_configs[leg] = {"lanes": lanes, "width": width}

        st.header("4. Vehicle Class Capabilities")
        vehicle_configs = {}
        for idx, v in enumerate(st.session_state.vehicle_types):
            cat = st.session_state.categories[idx] if idx < len(st.session_state.categories) else "Passenger"
            cl  = str(cat).lower()
            dm, dt = 60, 10
            if "rickshaw" in cl:   dm, dt = 30, 15
            elif "bicycle" in cl or "motorcycle" in cl: dm, dt = 60, 15
            elif "heavy" in cl or "truck" in cl:  dm, dt = 40, 6
            elif "bus" in cl:                     dm, dt = 50, 6
            cv = st.columns(3)
            with cv[0]: st.write(f"**{v}** ({cat})")
            with cv[1]: ms = st.number_input("Top Speed (km/h)", 10, 150, dm, 5, key=f"vmax_{idx}")
            with cv[2]: ts = st.number_input("Turn Speed (km/h)", 5, 50, dt, 1, key=f"vturn_{idx}")
            vehicle_configs[v] = {"max_speed": ms, "turn_speed": ts}

        if st.button("Save System Context", type="primary"):
            req_pts = set(["Intersection Center"] + [f"{l} Endpoint" for l in st.session_state.legs])
            if req_pts.issubset(set(st.session_state.coordinates.keys())):
                os.makedirs(pdata(proj), exist_ok=True)
                data = {
                    "vehicle_types": st.session_state.vehicle_types,
                    "pcus": st.session_state.pcus,
                    "categories": st.session_state.categories,
                    "movements": st.session_state.df_volume.to_dict(orient="records"),
                    "legs": st.session_state.legs,
                    "coordinates": st.session_state.coordinates,
                    "leg_configs": leg_configs,
                    "vehicle_configs": vehicle_configs
                }
                with open(cfg_path(proj), "w") as f: json.dump(data, f, indent=4)
                pstate["modules_completed"]["1_data_ingest"] = True
                save_pstate(proj, pstate)
                st.success("✅ Configuration saved!")
            else:
                st.error("⚠️ Map all coordinates first.")

    st.markdown("---")
    st.button("Next ➡ Trip Generation", type="secondary", on_click=go_to, args=("📈 2. Trip Generation & Forecasting",))

# ╔════════════════════════════════════════════════════════════╗
# ║  MODULE 2 — Trip Generation & Forecasting                 ║
# ╚════════════════════════════════════════════════════════════╝
elif choice == "📈 2. Trip Generation & Forecasting":
    st.title(f"Module 2: Trip Generation & Forecasting  —  {proj}")
    cp = cfg_path(proj)
    if not os.path.exists(cp):
        st.warning("Please complete Module 1 first."); st.stop()

    with open(cp) as f: config = json.load(f)
    vtypes, pcus, df_moves = config.get("vehicle_types",[]), config.get("pcus",[]), config.get("movements",[])
    legs_cfg = config.get("legs", [])

    st.header("1. Future Scenario Setup")
    agr = st.number_input("Annual Growth Rate (AGR %)", 0.0, 15.0, config.get("agr",5.0), 0.1)
    st.markdown("**(Optional) Trip Generation:** Enter total PCU volume injected by the mapped project.")
    total_trip_pcu = st.number_input("Total Generated Trips (PCU/Hr)", 0, 20000, 0, 50)

    total_base = sum(int(m.get(v,0)) * float(pcus[i] if i<len(pcus) else 1.0)
                     for m in df_moves for i,v in enumerate(vtypes))
    gen_trips  = {}
    if total_trip_pcu > 0 and total_base > 0:
        base_counts = {v: sum(int(m.get(v,0)) for m in df_moves) for v in vtypes}
        for v in vtypes:
            gen_trips[v] = int(round(total_trip_pcu * base_counts[v] / total_base))
    else:
        gen_trips = {v: 0 for v in vtypes}
    st.session_state.generated_trips = gen_trips

    if total_trip_pcu > 0:
        st.write("**Vehicle Breakdown (from baseline modal split):**")
        ct = st.columns(len(vtypes))
        for i,v in enumerate(vtypes):
            with ct[i]: st.metric(v, f"{gen_trips[v]} Veh")

    active_gen = sum(gen_trips.values())
    if active_gen > 0:
        if "Project Development Site" not in config.get("coordinates",{}):
            st.warning("⚠️ Map 'Project Development Site' in Module 1 for AI Trip Distribution.")
        else:
            st.markdown("#### 🤖 AI Trip Distribution Matrix")
            movements  = [m.get("Movement") for m in df_moves]
            if "ai_matrix" not in st.session_state or list(st.session_state.ai_matrix["Movement"]) != movements:
                st.session_state.ai_matrix = pd.DataFrame({"Movement": movements, "Percentage (%)": [0]*len(movements)})

            ac1, ac2 = st.columns([1,3])
            with ac1:
                if st.button("🤖 Auto-Distribute via AI", type="secondary", use_container_width=True):
                    if check_ollama_health():
                        crds  = config.get("coordinates",{})
                        proj_s = crds.get("Project Development Site",{})
                        ctr   = crds.get("Intersection Center",{})
                        legs_info = {l: crds.get(f"{l} Endpoint",{}) for l in legs_cfg}
                        pt = (f"You are a Traffic Engineer.\nProject at Lat:{proj_s.get('lat','?')} Lng:{proj_s.get('lng','?')}\n"
                              f"Intersection at Lat:{ctr.get('lat','?')} Lng:{ctr.get('lng','?')}\n"
                              f"Arms: " + ", ".join(f"{l}: {ep}" for l,ep in legs_info.items()) +
                              f"\nGenerate {active_gen} veh/hr. Movements:\n" + "\n".join(f"  - {mv}" for mv in movements) +
                              "\nReturn ONLY valid JSON mapping movement->integer percentage, summing to 100.")
                        with st.spinner("AI distributing trips…"):
                            try:
                                r = requests.post("http://127.0.0.1:11434/api/generate",
                                    json={"model":MODEL_NAME,"prompt":pt,"stream":False,"format":"json"}, timeout=120)
                                raw = r.json().get("response","")
                                s,e = raw.find("{"), raw.rfind("}")+1
                                dj  = json.loads(raw[s:e] if s>=0 and e>s else raw)
                                dn  = {k.strip():v for k,v in dj.items()}
                                np  = [int(dn.get(mv.strip(),0)) for mv in movements]
                                tot = sum(np)
                                if tot>0 and tot!=100: np[-1] += 100-tot
                                st.session_state.ai_matrix["Percentage (%)"] = np
                                st.success(f"✅ AI distributed {sum(np)}%")
                            except Exception as ex:
                                st.error(f"❌ AI failed: {ex}")
            with ac2:
                tp = int(st.session_state.ai_matrix["Percentage (%)"].sum())
                if tp==100: st.success(f"✅ Total: {tp}%")
                elif tp==0: st.info("Use AI or fill manually.")
                else: st.warning(f"⚠️ Total = {tp}% — must equal 100%")

            st.session_state.ai_matrix = st.data_editor(
                st.session_state.ai_matrix, use_container_width=True, hide_index=True,
                column_config={"Movement": st.column_config.TextColumn("Movement", disabled=True),
                               "Percentage (%)": st.column_config.NumberColumn("% Share", min_value=0, max_value=100, step=1)})

    if st.button("Save Forecast Context", type="primary"):
        config["agr"] = agr; config["generated_trips"] = gen_trips
        if "ai_matrix" in st.session_state:
            config["ai_matrix"] = st.session_state.ai_matrix.to_dict(orient="records")
        with open(cp, "w") as f: json.dump(config, f, indent=4)
        pstate["modules_completed"]["2_trip_gen"] = True
        save_pstate(proj, pstate)
        st.success("✅ Forecast settings saved!")

    st.markdown("---")
    st.button("Next ➡ Network Compiler", type="secondary", on_click=go_to, args=("⚙️ 3. Network Compiler",))

# ╔════════════════════════════════════════════════════════════╗
# ║  MODULE 3 — Network Compiler                              ║
# ╚════════════════════════════════════════════════════════════╝
elif choice == "⚙️ 3. Network Compiler":
    st.title(f"Module 3: Network Compiler  —  {proj}")

    cp = fcfg_path(proj) if os.path.exists(fcfg_path(proj)) else cfg_path(proj)
    if not os.path.exists(cp):
        st.warning("Please complete Module 1 first."); st.stop()

    st.info("Clicking the button below compiles **all 5 scenario folders** at once. "
            "Each scenario gets its own SUMO network and traffic route file.")

    # Show current compilation status
    if pstate["modules_completed"].get("3_network_compile"):
        st.success("✅ Network already compiled. You can recompile anytime to pick up changes.")
        sc = pstate.get("scenarios_compiled", {})
        status_rows = [{"Scenario": sname, "Folder": sslug,
                        "Status": "✅ Compiled" if sc.get(skey) else "❌ Failed"}
                       for skey, sname, sslug in SCENARIOS]
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    if st.button("🔨 Compile All 5 Scenarios", type="primary", use_container_width=False):
        with open(cp) as f: config = json.load(f)
        with st.spinner("Running netconvert and generating all 5 scenario folders…"):
            ok, results = compile_all_scenarios(config, proj)
        if ok:
            st.success("✅ All 5 scenarios compiled successfully!")
        else:
            st.warning("⚠️ Some scenarios failed to compile:")
        res_rows = [{"Scenario": SCENARIOS[i][1], "Status": "✅ OK" if results[SCENARIOS[i][0]][0] else f"❌ {results[SCENARIOS[i][0]][1]}"}
                    for i in range(5)]
        st.dataframe(pd.DataFrame(res_rows), use_container_width=True, hide_index=True)
        pstate = load_pstate(proj)  # reload updated state
        st.rerun()

    st.markdown("---")
    st.button("Next ➡ AI Orchestration", type="secondary", on_click=go_to, args=("🤖 4. AI Orchestration",))

# ╔════════════════════════════════════════════════════════════╗
# ║  MODULE 4 — AI Orchestration                              ║
# ╚════════════════════════════════════════════════════════════╝
elif choice == "🤖 4. AI Orchestration":
    st.title(f"Module 4: Local AI Orchestration  —  {proj}")
    if not check_ollama_health(): st.stop()
    cp = cfg_path(proj)
    if not os.path.exists(cp): st.warning("Module 1 config missing."); st.stop()

    if "chat_history" not in st.session_state:
        with open(cp) as f: ctx = f.read()
        st.session_state.chat_history = [{"role":"system","content":
            f"Optimize the following intersection parameters and return JSON:\n{ctx}\n"
            "Respond ONLY with valid JSON using keys: action ('ask' or 'finalize'), message, optimized_parameters."}]
        with st.spinner("🤖 Analyzing Network…"):
            res = call_ollama(st.session_state.chat_history)
            try: st.session_state.chat_history.append({"role":"assistant","content":res,"parsed":json.loads(res)})
            except: pass

    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"]=="assistant" and "parsed" in msg:
            pd_ = msg["parsed"]
            if pd_.get("action")=="error": st.error(pd_.get("message"))
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(pd_.get("message","Analyzing…"))
                    if pd_.get("action")=="finalize":
                        with st.expander("Finalized AI Context"): st.json(pd_.get("optimized_parameters",{}))
                        if st.button("Commit to SUMO", type="primary", key=f"c_{i}"):
                            with open(cp) as f: fc = json.load(f)
                            # Protect structural fields — AI must not overwrite these
                            PROTECTED = {"legs","leg_configs","coordinates","vehicle_types",
                                         "pcus","categories","vehicle_configs","movements"}
                            ai_params = pd_.get("optimized_parameters", {})
                            safe_params = {k: v for k, v in ai_params.items() if k not in PROTECTED}
                            fc.update(safe_params)
                            os.makedirs(pdata(proj), exist_ok=True)
                            with open(fcfg_path(proj),"w") as f: json.dump(fc,f,indent=4)
                            pstate["modules_completed"]["4_ai_orchestration"] = True
                            save_pstate(proj, pstate)
                            st.success("✅ Saved. Return to Module 3 to recompile.")
        elif msg["role"]=="user":
            with st.chat_message("user", avatar="👤"): st.write(msg["content"])

    if prompt := st.chat_input("Clarify constraints…"):
        st.session_state.chat_history.append({"role":"user","content":prompt})
        with st.spinner("🤖 Retweaking parameters…"):
            res = call_ollama(st.session_state.chat_history)
            try: st.session_state.chat_history.append({"role":"assistant","content":res,"parsed":json.loads(res)})
            except: pass
        st.rerun()

    st.markdown("---")
    st.button("Next ➡ Live Dashboard", type="secondary", on_click=go_to, args=("🚦 5. Live Dashboard",))

# ╔════════════════════════════════════════════════════════════╗
# ║  MODULE 5 — Live Dashboard (5 independent scenarios)      ║
# ╚════════════════════════════════════════════════════════════╝
elif choice == "🚦 5. Live Dashboard":
    st.title(f"Module 5: SUMO Live Dashboard  —  {proj}")
    pstate = load_pstate(proj)

    if not pstate["modules_completed"].get("3_network_compile"):
        st.warning("⚠️ Please compile the network (Module 3) first.")
        st.stop()

    # Always read structural fields from module_a (base config).
    # module_c (AI-committed) only contributes non-structural optimised params.
    base_cp = cfg_path(proj)
    if not os.path.exists(base_cp):
        st.error("Base configuration not found. Please complete Module 1."); st.stop()
    with open(base_cp) as f: base_conf = json.load(f)

    # Merge AI params on top if available, but keep structural fields from base
    conf = dict(base_conf)
    ai_cp = fcfg_path(proj)
    if os.path.exists(ai_cp):
        with open(ai_cp) as f: ai_conf = json.load(f)
        PROTECTED = {"legs","leg_configs","coordinates","vehicle_types",
                     "pcus","categories","vehicle_configs","movements"}
        for k, v in ai_conf.items():
            if k not in PROTECTED:
                conf[k] = v

    legs       = conf.get("legs", [])
    leg_configs = conf.get("leg_configs", {})

    # Defensive: if legs contains dicts (corrupted AI overwrite), recover from leg_configs
    if legs and not isinstance(legs[0], str):
        legs = sorted(leg_configs.keys())

    # Initialise per-scenario sim_states dict
    if "sim_states" not in st.session_state:
        st.session_state.sim_states = {}

    def _init_sim_state(skey):
        lf = "|".join(sorted(legs))
        st.session_state.sim_states[skey] = {
            "running": False, "paused": False, "simulation_finished": False,
            "step": 0, "total_inserted": 0, "total_arrived": 0, "avg_speed": 0.0,
            "error": None, "_legs_fp": lf,
            "legs": {k: {"vol":0,"queue":0,"delay_avg":0.0,"los":"A","max_q":0} for k in legs}
        }

    # Pre-load any saved results from project.json into sim_states
    saved_results = pstate.get("scenario_results", {})
    for skey, sname, sslug in SCENARIOS:
        if skey not in st.session_state.sim_states:
            if pstate["scenarios_simulated"].get(skey) and skey in saved_results:
                # Restore finished state from saved results
                lf = "|".join(sorted(legs))
                st.session_state.sim_states[skey] = {
                    "running": False, "paused": False, "simulation_finished": True,
                    "step": 3600, "total_inserted": 0, "total_arrived": 0, "avg_speed": 0.0,
                    "error": None, "_legs_fp": lf,
                    "legs": {k: {
                        "vol":      saved_results[skey].get(k,{}).get("vol",0),
                        "queue":    0,
                        "delay_avg":saved_results[skey].get(k,{}).get("delay_avg",0.0),
                        "los":      saved_results[skey].get(k,{}).get("los","A"),
                        "max_q":    saved_results[skey].get(k,{}).get("max_q",0),
                    } for k in legs}
                }
            else:
                _init_sim_state(skey)

    # ── 5 Tabs ──
    tab_labels = [f"{'✅' if pstate['scenarios_simulated'].get(s[0]) else '⬜'} {s[1]}" for s in SCENARIOS]
    tabs = st.tabs(tab_labels)

    for tab_i, (skey, sname, sslug) in enumerate(SCENARIOS):
        with tabs[tab_i]:
            sd  = pscene(proj, sslug)
            cfg_f = os.path.join(sd, "sim.sumocfg")
            compiled = pstate["scenarios_compiled"].get(skey, False)

            if not compiled or not os.path.exists(cfg_f):
                st.warning(f"⚠️ **{sname}** not yet compiled. Run Module 3 first.")
                continue

            as_ = st.session_state.sim_states[skey]

            ctrl, panel = st.columns([1, 4])
            with ctrl:
                if not as_["running"] and not as_["simulation_finished"]:
                    if st.button(f"▶ Launch", type="primary", key=f"run_{skey}", use_container_width=True):
                        as_["running"] = True
                        start_traci_thread(as_, legs, leg_configs, cfg_f, skey)
                        st.rerun()

                if as_["running"]:
                    if st.button("⏸ Pause/Resume", key=f"pause_{skey}", use_container_width=True):
                        as_["paused"] = not as_["paused"]
                    if st.button("⏹ Stop", key=f"stop_{skey}", use_container_width=True, type="secondary"):
                        as_["running"] = False; as_["simulation_finished"] = True

                if as_["simulation_finished"]:
                    if st.button("🔄 Re-run", key=f"rerun_{skey}", use_container_width=True):
                        _init_sim_state(skey); st.rerun()

                st.markdown("---")
                st.metric("Time",     f"{as_['step']}s")
                st.metric("Inserted", as_["total_inserted"])
                st.metric("Arrived",  as_["total_arrived"])
                st.metric("Avg Spd",  f"{as_['avg_speed']:.1f} km/h")

            with panel:
                st.subheader(f"🚥 {sname} — Approach Leg Metrics")
                if as_.get("error"):
                    st.error(f"TraCI Error: {as_['error']}")

                rows = []
                for leg in legs:
                    d  = as_["legs"][leg]
                    ln = leg_configs.get(leg,{}).get("lanes",2)
                    cap= ln * 900
                    st_val = max(1, as_["step"])
                    if as_["simulation_finished"]:
                        vc = round(d["vol"]/cap,3) if cap else 0
                        rows.append({"Approach Leg":leg,"Final Vol (vph)":d["vol"],
                                     "Max Queue/Lane (m)":round(d["max_q"],1),
                                     "Avg Delay (s)":round(d["delay_avg"],1),
                                     "V/C Ratio":vc,"LOS":d["los"]})
                    else:
                        vc = round((d["vol"]/st_val)*3600/cap,3) if cap and st_val>60 else 0
                        rows.append({"Approach Leg":leg,"Vol (live)":d["vol"],
                                     "Queue (m)":round(d["queue"],1),
                                     "Avg Delay (s)":round(d["delay_avg"],1),
                                     "V/C (Proj)":vc,"LOS":d["los"]})

                df_disp = pd.DataFrame(rows)
                st.dataframe(df_disp, use_container_width=True, hide_index=True)

                if as_["simulation_finished"]:
                    st.success(f"✅ {sname} simulation complete.")
                    # Save results to project.json
                    res_entry = {}
                    for leg in legs:
                        d   = as_["legs"][leg]
                        cap = leg_configs.get(leg,{}).get("lanes",2) * 900
                        res_entry[leg] = {"vol": d["vol"], "delay_avg": round(d["delay_avg"],1),
                                          "max_q": round(d["max_q"],1),
                                          "vc_ratio": round(d["vol"]/cap,3) if cap else 0,
                                          "los": d["los"]}
                    ps_fresh = load_pstate(proj)
                    if not ps_fresh["scenarios_simulated"].get(skey):
                        ps_fresh["scenarios_simulated"][skey]   = True
                        ps_fresh["scenario_results"][skey]      = res_entry
                        save_pstate(proj, ps_fresh)

                    # CSV export
                    csv_ = df_disp.to_csv(index=False).encode("utf-8")
                    safe = sname.replace(" ","_").replace("(","").replace(")","").replace("-","_")
                    st.download_button(f"📥 Export {sname} Report", csv_,
                                       file_name=f"{proj}_{safe}_report.csv", mime="text/csv",
                                       key=f"dl_{skey}")

    # ── Auto-refresh if any scenario is still running ──
    any_running = any(
        st.session_state.sim_states.get(skey, {}).get("running", False)
        for skey, _, _ in SCENARIOS
    )
    if any_running:
        time.sleep(0.5)
        st.rerun()

# ╔════════════════════════════════════════════════════════════╗
# ║  MODULE 6 — Performance Comparison Report                   ║
# ╚════════════════════════════════════════════════════════════╝
elif choice == "📊 6. Comparison Report":
    st.title(f"Module 6: Performance Comparison Report  —  {proj}")
    pstate   = load_pstate(proj)
    results  = pstate.get("scenario_results", {})

    # ── Load base config for leg geometry ───────────────────────────
    base_cp = cfg_path(proj)
    if os.path.exists(base_cp):
        with open(base_cp) as f: bc = json.load(f)
        coords   = bc.get("coordinates", {})
        legs     = bc.get("legs", [])
        leg_cfgs = bc.get("leg_configs", {})
    else:
        coords, legs, leg_cfgs = {}, [], {}

    # ── Arm length = max haversine distance from center to any endpoint ──
    def haversine(lat1, lon1, lat2, lon2):
        R    = 6371000.0
        phi1 = math.radians(lat1); phi2 = math.radians(lat2)
        dl   = math.radians(lon2 - lon1); dp = math.radians(lat2 - lat1)
        a    = math.sin(dp/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    ic  = coords.get("Intersection Center", {})
    arm = 0
    for leg in legs:
        ep = coords.get(f"{leg} Endpoint", {})
        if ic and ep:
            arm = max(arm, haversine(ic["lat"], ic["lng"], ep["lat"], ep["lng"]))
    arm_len = int(round(arm))   # metres

    # ── Helper functions ─────────────────────────────────────────
    def max_val(skey, metric, dec=3):
        if skey not in results: return None
        vals = [v.get(metric, 0) for v in results[skey].values() if isinstance(v, dict)]
        return round(max(vals), dec) if vals else None

    def max_los(skey):
        if skey not in results: return "-"
        best_los, best_delay = "A", -1.0
        for v in results[skey].values():
            if isinstance(v, dict) and v.get("delay_avg", 0) > best_delay:
                best_delay = v["delay_avg"]; best_los = v.get("los", "A")
        return best_los

    def fmt(v, dec):
        if v is None: return "-"
        return str(int(v)) if dec == 0 else str(round(v, dec))

    def diff_str(b, c, dec):
        if b is None or c is None: return "-"
        d = round(c - b, dec)
        return f"+{d}" if d > 0 else str(d)

    def los_diff(los_b, los_c):
        if los_b == "-" or los_c == "-": return "-"
        return "No Change" if los_b == los_c else f"{los_b}→{los_c}"

    # ── Column header labels ─────────────────────────────────────────
    H = [
        "Performance Indicator",
        "Base Year (Existing) (a)",
        "After 5 Years (Without Project) (b)",
        "After 5 Years (With Project) (c)",
        "Difference (Year 5) d = c−b",
        "After 20 Years (Without Project) (e)",
        "After 20 Years (With Project) (f)",
        "Difference (Year 20) g = f−e",
    ]

    # Short names map: (skey, metric, decimals)
    # present=a, 5yr_no_dev=b, 5yr_with_dev=c, 20yr_no_dev=e, 20yr_with_dev=f
    rows = []

    # ── Row 1: V/C Ratio ──
    a=max_val("present","vc_ratio",3); b=max_val("5yr_no_dev","vc_ratio",3)
    c=max_val("5yr_with_dev","vc_ratio",3); e=max_val("20yr_no_dev","vc_ratio",3); f=max_val("20yr_with_dev","vc_ratio",3)
    rows.append({H[0]:"V/C Ratio",H[1]:fmt(a,3),H[2]:fmt(b,3),H[3]:fmt(c,3),
                 H[4]:diff_str(b,c,3),H[5]:fmt(e,3),H[6]:fmt(f,3),H[7]:diff_str(e,f,3)})

    # ── Row 2: Delay (sec) ──
    a=max_val("present","delay_avg",1); b=max_val("5yr_no_dev","delay_avg",1)
    c=max_val("5yr_with_dev","delay_avg",1); e=max_val("20yr_no_dev","delay_avg",1); f=max_val("20yr_with_dev","delay_avg",1)
    rows.append({H[0]:"Delay (sec)",H[1]:fmt(a,1),H[2]:fmt(b,1),H[3]:fmt(c,1),
                 H[4]:diff_str(b,c,1),H[5]:fmt(e,1),H[6]:fmt(f,1),H[7]:diff_str(e,f,1)})

    # ── Row 3: Queue Length (m) ──
    a=max_val("present","max_q",1); b=max_val("5yr_no_dev","max_q",1)
    c=max_val("5yr_with_dev","max_q",1); e=max_val("20yr_no_dev","max_q",1); f=max_val("20yr_with_dev","max_q",1)
    rows.append({H[0]:"Queue Length (m)",H[1]:fmt(a,1),H[2]:fmt(b,1),H[3]:fmt(c,1),
                 H[4]:diff_str(b,c,1),H[5]:fmt(e,1),H[6]:fmt(f,1),H[7]:diff_str(e,f,1)})

    # ── Row 4: Lane Length (m) — fixed geometry, same for all, no diff ──
    al = str(arm_len) if arm_len > 0 else "-"
    rows.append({H[0]:"Lane Length (m)",H[1]:al,H[2]:al,H[3]:al,
                 H[4]:"-",H[5]:al,H[6]:al,H[7]:"-"})

    # ── Row 5: Level of Service ──
    la=max_los("present"); lb=max_los("5yr_no_dev"); lc=max_los("5yr_with_dev")
    le=max_los("20yr_no_dev"); lf=max_los("20yr_with_dev")
    rows.append({H[0]:"Level of Service",H[1]:la,H[2]:lb,H[3]:lc,
                 H[4]:los_diff(lb,lc),H[5]:le,H[6]:lf,H[7]:los_diff(le,lf)})

    # ── Display ───────────────────────────────────────────
    df_cmp = pd.DataFrame(rows).set_index(H[0])

    # Highlight difference columns
    def highlight_diff(s):
        styles = []
        for v in s:
            if isinstance(v, str) and v.startswith("+"):
                styles.append("color: #e74c3c; font-weight: bold")
            elif isinstance(v, str) and v.startswith("-") and v != "-":
                styles.append("color: #27ae60; font-weight: bold")
            elif v == "No Change":
                styles.append("color: #7f8c8d")
            else:
                styles.append("")
        return styles

    diff_cols = [H[4], H[7]]
    styled = df_cmp.style.apply(highlight_diff, subset=diff_cols)
    st.dataframe(styled, use_container_width=True)

    # ── Legend ──
    st.caption("🟥 Red = worsening with project  |  🟢 Green = improving with project")

    # ── Missing scenarios note ──────────────────────────────────
    available_keys = set(results.keys())
    missing = [sname for skey, sname, _ in SCENARIOS if skey not in available_keys]
    if missing:
        st.info(f"ℹ️ Not yet simulated: **{', '.join(missing)}**. Run them in Module 5 to fill in the table.")

    # ── Export ────────────────────────────────────────────
    st.markdown("---")
    csv_out = df_cmp.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Comparison Report (CSV)",
        data=csv_out,
        file_name=f"{proj}_comparison_report.csv",
        mime="text/csv"
    )
