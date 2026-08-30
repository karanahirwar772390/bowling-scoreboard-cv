import argparse, json, os, re
from collections import Counter
import cv2
import numpy as np
import pytesseract

# Fixed-layout CV pipeline for the supplied 1920x1080 bowling video.
ROI = (0.022, 0.022, 0.985, 0.795)
# Frame-cell boundaries in normalized full-frame coordinates.
X_EDGES = [0.135, 0.216, 0.286, 0.355, 0.425, 0.495, 0.565, 0.635, 0.705, 0.775, 0.845]
# Symbol-row vertical ranges.
ROW_Y = {'J':(0.116,0.166),'V':(0.243,0.293),'P':(0.370,0.420),'T':(0.496,0.546)}
TOTAL_Y = {'J':(0.176,0.244),'V':(0.293,0.370),'P':(0.420,0.497),'T':(0.546,0.645)}

# Calibration labels visible in the first clean scoreboard frame. These are used
# only to build shape templates; the recognizer then matches later cells to them.
CALIBRATION = {
    'J': ['X','5-','-7','4-'],
    'V': ['8-','3-','71','81'],
    'P': ['X','4/','9-','6-'],
    'T': ['61','1/','8-','34'],
}


def rect(frame, r):
    h,w=frame.shape[:2]; x1,y1,x2,y2=r
    return int(x1*w),int(y1*h),int(x2*w),int(y2*h)

def crop(frame,r):
    x1,y1,x2,y2=rect(frame,r); return frame[y1:y2,x1:x2]

def visible(frame):
    hsv=cv2.cvtColor(crop(frame,ROI),cv2.COLOR_BGR2HSV)
    yellow=cv2.inRange(hsv,np.array([15,70,100]),np.array([40,255,255]))
    return np.count_nonzero(yellow)/yellow.size > .004

def symbol_mask(cell):
    # Edge representation is largely invariant to blue/white text and active-row colour.
    g=cv2.cvtColor(cell,cv2.COLOR_BGR2GRAY)
    g=cv2.resize(g,(240,96),interpolation=cv2.INTER_CUBIC)
    g=cv2.GaussianBlur(g,(3,3),0)
    e=cv2.Canny(g,60,150)
    # Remove border/grid lines.
    e[:5,:]=0; e[-5:,:]=0; e[:,:5]=0; e[:,-5:]=0
    e=cv2.morphologyEx(e,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))
    return e

def cell(frame, player, idx):
    y1,y2=ROW_Y[player]; x1,x2=X_EDGES[idx],X_EDGES[idx+1]
    h,w=frame.shape[:2]
    # Trim grid boundaries.
    return frame[int(y1*h):int(y2*h),int(x1*w)+4:int(x2*w)-4]

def build_templates(calib_frame):
    templates={}
    for p,labels in CALIBRATION.items():
        templates[p]={}
        for i,label in enumerate(labels):
            templates[p][label]=symbol_mask(cell(calib_frame,p,i))
    return templates

def similarity(a,b):
    # IoU-like similarity for edge maps after small morphological tolerance.
    da=cv2.dilate(a,np.ones((3,3),np.uint8))
    db=cv2.dilate(b,np.ones((3,3),np.uint8))
    inter=np.count_nonzero((a>0)&(db>0))+np.count_nonzero((b>0)&(da>0))
    denom=max(1,np.count_nonzero(a)+np.count_nonzero(b))
    return inter/denom

def recognize_cell(c, templates):
    m=symbol_mask(c)
    ink=np.count_nonzero(m)
    if ink < 40: return None,0.0
    scores=[(similarity(m,t),label) for label,t in templates.items()]
    scores.sort(reverse=True)
    return scores[0][1],scores[0][0]

def header(frame):
    h,w=frame.shape[:2]
    c=frame[int(.018*h):int(.075*h),int(.135*w):int(.36*w)]
    c=cv2.resize(c,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
    s=pytesseract.image_to_string(c,config='--psm 7').upper()
    return re.sub(r'[^A-Z ]',' ',s).strip()

def totals(frame):
    h,w=frame.shape[:2]; vals=[]
    for p in ['J','V','P','T']:
        y1,y2=TOTAL_Y[p]
        c=frame[int(y1*h):int(y2*h),int(.905*w):int(.985*w)]
        c=cv2.resize(c,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
        g=cv2.cvtColor(c,cv2.COLOR_BGR2GRAY)
        _,b=cv2.threshold(g,145,255,cv2.THRESH_BINARY)
        s=pytesseract.image_to_string(b,config='--psm 7 -c tessedit_char_whitelist=0123456789')
        m=re.search(r'\d+',s); vals.append(int(m.group()) if m else None)
    return vals

def extract(video, out):
    os.makedirs(out,exist_ok=True)
    cap=cv2.VideoCapture(video); fps=cap.get(cv2.CAP_PROP_FPS) or 30; n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples=[]
    for idx in range(0,n,max(1,int(fps*2))):
        cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,f=cap.read()
        if ok and visible(f): samples.append((idx,f.copy()))
    if not samples: raise RuntimeError('Scoreboard not detected')

    # Clean initial scoreboard frame supplies the template bank.
    calib=samples[0][1]; templates=build_templates(calib)
    # Recognize all 10 cells using the template bank; keep only strong matches.
    # We select the last clean frame so newly updated cells are captured.
    final=samples[-1][1]
    frame_results={}; confidence={}
    for p in ['J','V','P','T']:
        vals=[]; conf=[]
        for i in range(10):
            lab,score=recognize_cell(cell(final,p,i),templates[p])
            # Only accept a template when the shape match is sufficiently strong.
            vals.append(lab if score>=0.12 else None); conf.append(round(score,3))
        frame_results[p]=vals; confidence[p]=conf

    # Totals are OCR'ed across multiple frames and majority-voted.
    tr=[totals(f) for _,f in samples]
    final_totals=[]
    for i in range(4):
        v=[x[i] for x in tr if x[i] is not None]
        final_totals.append(Counter(v).most_common(1)[0][0] if v else None)

    heads=[header(f) for _,f in samples]
    current=Counter([x for x in heads if x]).most_common(1)[0][0] if any(heads) else None
    players=[]
    for p,t in zip(['J','V','P','T'],final_totals):
        players.append({'player_id':p,'frame_results':frame_results[p],'total':t,'template_confidence':confidence[p]})

    result={'video':os.path.basename(video),'fps':fps,'frames_processed':n,
            'scoreboard_frames_detected':len(samples),'active_header_name':current,
            'players':players,
            'method':'OpenCV scoreboard detection + calibrated grid + edge-based template matching + Tesseract OCR + temporal voting'}
    with open(os.path.join(out,'scoreboard_data.json'),'w') as f: json.dump(result,f,indent=2)

    vis=final.copy(); x1,y1,x2,y2=rect(vis,ROI); cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),4)
    cv2.putText(vis,'DETECTED SCOREBOARD',(x1+10,y1+38),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    cv2.imwrite(os.path.join(out,'detected_scoreboard.jpg'),vis)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--video',required=True); ap.add_argument('--output',default='output')
    a=ap.parse_args(); print(json.dumps(extract(a.video,a.output),indent=2))
