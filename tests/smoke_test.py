from web_app.utils.mapping import load_mapping, detector_index_to_project, get_thresholds
import json

def run():
    cfg = load_mapping()
    print("Mapping loaded:", json.dumps(cfg, indent=2))
    thr, nms = get_thresholds()
    print(f"confidence_threshold={thr}, nms_iou={nms}")
    sample = ['0','2','3','5','7','99']
    for s in sample:
        print(s, "->", detector_index_to_project(s))
    print("Smoke test OK")

if __name__ == '__main__':
    run()
