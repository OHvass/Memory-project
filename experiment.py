import csv
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

# Paths are always relative to the folder containing this script.
BASE = Path(__file__).parent.absolute()
IMGDIR = BASE / "images"
RESDIR = BASE / "results"
PROGRESS = BASE / "image_progress.json"
IMGDIR.mkdir(parents=True, exist_ok=True)
RESDIR.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGES_PER_TRIAL = 15
STANDARD_SECONDS = 2.0
FAST_SECONDS = 1.0
DELAY_SECONDS = 15

CONDITIONS = {
    "standard": {
        "name": "Standardforsøg",
        "short": "Standard",
        "description": "15 billeder · 2,0 sek. pr. billede · recall med det samme",
    },
    "fast": {
        "name": "Hurtigere visninger",
        "short": "Hurtig",
        "description": "15 billeder · 1,0 sek. pr. billede · recall med det samme",
    },
    "arithmetic": {
        "name": "Plus/minus i 15 sekunder",
        "short": "Plus/minus",
        "description": "15 billeder · 2,0 sek. pr. billede · regnestykker i 15 sek. · recall",
    },
    "pause": {
        "name": "Almindelig pause i 15 sekunder",
        "short": "Pause",
        "description": "15 billeder · 2,0 sek. pr. billede · tom pause i 15 sek. · recall",
    },
}


def natural_key(s):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def label_from_path(p):
    s = re.sub(r"^\d+[_\-\s]*", "", p.stem)
    return re.sub(r"\s+", " ", s.replace("_", " ").replace("-", " ")).strip()


def all_images():
    files = [p for p in IMGDIR.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
    files.sort(key=lambda p: natural_key(str(p.relative_to(IMGDIR))))
    return files


def get_next_index():
    try:
        return max(0, int(json.loads(PROGRESS.read_text(encoding="utf-8"))["next_image_index"]))
    except Exception:
        return 0


def set_next_index(i):
    PROGRESS.write_text(json.dumps({"next_image_index": i}, indent=2), encoding="utf-8")


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sample_sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))


def tcrit95_df(df):
    # Good-enough two-sided 95% critical values for this teaching experiment.
    table = {
        1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,
        9:2.262,10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,
        16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,21:2.080,22:2.074,
        23:2.069,24:2.064,25:2.060,26:2.056,27:2.052,28:2.048,29:2.045,
        30:2.042,40:2.021,60:2.000,120:1.980
    }
    if df <= 1: return table[1]
    ks = sorted(table)
    for k in ks:
        if df <= k: return table[k]
    return 1.96


def ci95(xs, clamp=False):
    n = len(xs)
    if n < 2:
        return (float("nan"), float("nan"))
    m = mean(xs)
    sd = sample_sd(xs)
    se = sd / math.sqrt(n)
    d = tcrit95_df(n-1) * se
    lo, hi = m-d, m+d
    if clamp:
        lo, hi = max(0,lo), min(1,hi)
    return lo, hi


def diff_ci95(xs, ys):
    # Difference of independent condition means: mean(xs) - mean(ys), Welch-style CI.
    if len(xs) < 2 or len(ys) < 2:
        return mean(xs)-mean(ys) if xs and ys else float("nan"), float("nan"), float("nan")
    mx, my = mean(xs), mean(ys)
    vx, vy = sample_sd(xs)**2, sample_sd(ys)**2
    a, b = vx/len(xs), vy/len(ys)
    se2 = a+b
    if se2 <= 0:
        return mx-my, mx-my, mx-my
    df_num = se2**2
    df_den = (a*a/(len(xs)-1) if len(xs)>1 else 0) + (b*b/(len(ys)-1) if len(ys)>1 else 0)
    df = df_num/df_den if df_den > 0 else min(len(xs),len(ys))-1
    d = tcrit95_df(df) * math.sqrt(se2)
    return mx-my, (mx-my)-d, (mx-my)+d


def pct(x):
    return "—" if x is None or math.isnan(x) else f"{100*x:.1f}%"


def pp(x):
    return "—" if x is None or math.isnan(x) else f"{100*x:+.1f} pp"


def ci_text(lo, hi, effect=False):
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi): return "—"
    return f"[{100*lo:+.1f}, {100*hi:+.1f}] pp" if effect else f"[{100*lo:.1f}%, {100*hi:.1f}%]"


class FreeRecallApp:
    def __init__(self, root):
        self.root = root
        root.title("Free Recall – 4 forsøg")
        root.configure(bg="white")
        root.attributes("-fullscreen", True)
        root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))
        root.protocol("WM_DELETE_WINDOW", self.quit_program)

        self.files = all_images()
        if len(self.files) < IMAGES_PER_TRIAL:
            raise RuntimeError(f"Fandt kun {len(self.files)} billeder i images-mappen. Der kræves mindst {IMAGES_PER_TRIAL}.")

        self.idx = get_next_index()
        if self.idx >= len(self.files):
            self.idx = 0
            set_next_index(0)

        pid = simpledialog.askstring("Deltager-ID", "Indtast deltager-ID (fx P01):", parent=root)
        if not pid:
            root.destroy(); return
        self.pid = re.sub(r"[^A-Za-z0-9_-]+", "_", pid.strip())
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.items_file = RESDIR / f"{self.pid}_{self.session}_items.csv"
        self.trials_file = RESDIR / f"{self.pid}_{self.session}_trials.csv"
        self._init_csvs()

        self.trial = 0
        self.condition = None
        self.current = []
        self.photo = None
        self.score_photos = []
        self.score_vars = []
        self.raw_response = ""
        self.dynamic = []
        self.math_attempts = 0
        self.math_correct = 0
        self.math_answer = None
        self.math_after_id = None

        self.frame = tk.Frame(root, bg="white")
        self.frame.pack(fill="both", expand=True)
        self.main_text = tk.Label(self.frame, bg="white", fg="black", font=("Arial", 28), justify="center", wraplength=1200)
        self.main_text.pack(expand=True)
        self.image_label = tk.Label(self.frame, bg="white")
        self.image_label.pack(expand=True)
        self.show_menu()

    def _init_csvs(self):
        with open(self.items_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "participant","session","trial","condition","global_image","serial_position",
                "filename","label","recalled_manual","raw_response"
            ])
        with open(self.trials_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "participant","session","trial","condition","first_global_image","last_global_image",
                "n_recalled","recall_rate","primacy_recall","middle_recall","recency_recall",
                "primacy_effect","recency_effect","math_attempts","math_correct","raw_response"
            ])

    def clear(self):
        if self.math_after_id:
            try: self.root.after_cancel(self.math_after_id)
            except Exception: pass
            self.math_after_id = None
        for w in self.dynamic:
            try: w.destroy()
            except Exception: pass
        self.dynamic = []
        self.main_text.config(text="")
        self.image_label.config(image="")
        self.photo = None

    def show_menu(self):
        self.clear()
        self.root.unbind("<space>")
        if len(self.files)-self.idx < IMAGES_PER_TRIAL:
            self.main_text.config(text="Der er færre end 15 ubrugte billeder tilbage.\n\nTilføj flere billeder eller nulstil image_progress.json.")
            return

        rows = self.load_all_trials()
        counts = {k: sum(r["condition"]==k for r in rows) for k in CONDITIONS}
        self.main_text.config(text="FREE RECALL\n\nVælg hvilket forsøg du vil køre")

        box = tk.Frame(self.frame, bg="white")
        box.pack(pady=(0,15)); self.dynamic.append(box)
        for key in ["standard","fast","arithmetic","pause"]:
            c = CONDITIONS[key]
            b = tk.Button(box, text=f"{c['name']}\n{c['description']}\nGemte trials: {counts[key]}",
                          font=("Arial",16), width=54, pady=10,
                          command=lambda k=key: self.prepare_trial(k))
            b.pack(pady=5)

        btns = tk.Frame(self.frame,bg="white"); btns.pack(pady=8); self.dynamic.append(btns)
        tk.Button(btns,text="Se al statistik",font=("Arial",17,"bold"),command=self.show_analysis).pack(side="left",padx=8)
        tk.Button(btns,text="Luk",font=("Arial",15),command=self.quit_program).pack(side="left",padx=8)

    def prepare_trial(self, condition):
        self.condition = condition
        self.clear()
        c = CONDITIONS[condition]
        self.main_text.config(text=f"{c['name']}\n\n{c['description']}\n\nTryk MELLEMRUM for at starte.")
        self.root.bind("<space>", self.start_trial)

    def start_trial(self, event=None):
        self.root.unbind("<space>")
        if len(self.files)-self.idx < IMAGES_PER_TRIAL:
            self.show_menu(); return
        self.trial += 1
        self.current = self.files[self.idx:self.idx+IMAGES_PER_TRIAL]
        self.pos = -1
        self.math_attempts = 0; self.math_correct = 0
        self.clear(); self.show_next_image()

    def show_next_image(self):
        self.pos += 1
        if self.pos >= IMAGES_PER_TRIAL:
            if self.condition == "arithmetic": self.show_arithmetic()
            elif self.condition == "pause": self.show_pause()
            else: self.show_recall()
            return
        im = Image.open(self.current[self.pos]).convert("RGB")
        im.thumbnail((900,650))
        self.photo = ImageTk.PhotoImage(im)
        self.image_label.config(image=self.photo)
        seconds = FAST_SECONDS if self.condition=="fast" else STANDARD_SECONDS
        self.root.after(int(seconds*1000), self.show_next_image)

    def show_pause(self):
        self.clear()
        # Deliberately blank: no secondary task during the 15 s control delay.
        self.main_text.config(text="")
        self.root.after(DELAY_SECONDS*1000, self.show_recall)

    def make_math_problem(self):
        op = random.choice(["+","−"])
        if op == "+":
            a = random.randint(11,69); b = random.randint(2,19); ans = a+b
        else:
            b = random.randint(2,19); a = random.randint(max(11,b+1),79); ans = a-b
        return f"{a} {op} {b} = ?", ans

    def show_arithmetic(self):
        self.clear()
        panel = tk.Frame(self.frame,bg="white"); panel.pack(expand=True); self.dynamic.append(panel)
        tk.Label(panel,text="Regn så mange stykker som muligt",font=("Arial",24),bg="white").pack(pady=15)
        self.math_label = tk.Label(panel,text="",font=("Arial",44,"bold"),bg="white"); self.math_label.pack(pady=20)
        self.math_entry = tk.Entry(panel,font=("Arial",30),width=8,justify="center"); self.math_entry.pack(pady=10); self.math_entry.focus_set()
        tk.Label(panel,text="Skriv svaret og tryk Enter",font=("Arial",15),bg="white").pack(pady=8)
        self.math_entry.bind("<Return>", self.submit_math)
        self.next_math_problem()
        self.math_after_id = self.root.after(DELAY_SECONDS*1000, self.finish_arithmetic)

    def next_math_problem(self):
        q, self.math_answer = self.make_math_problem()
        self.math_label.config(text=q)
        self.math_entry.delete(0,"end")

    def submit_math(self,event=None):
        try:
            ans = int(self.math_entry.get().strip())
        except Exception:
            return
        self.math_attempts += 1
        if ans == self.math_answer: self.math_correct += 1
        self.next_math_problem()

    def finish_arithmetic(self):
        self.math_after_id = None
        self.show_recall()

    def show_recall(self):
        self.clear()
        self.main_text.config(text="Skriv alle de ting, du kan huske.\nRækkefølgen er ligegyldig. Adskil svar med komma.")
        entry = tk.Text(self.frame,height=6,width=70,font=("Arial",20),wrap="word"); entry.pack(pady=20); entry.focus_set()
        button = tk.Button(self.frame,text="Fortsæt til scoring",font=("Arial",18),command=lambda:self.show_scoring(entry)); button.pack(pady=10)
        self.dynamic=[entry,button]

    def show_scoring(self, entry):
        self.raw_response = entry.get("1.0","end").strip()
        self.clear(); self.score_vars=[]; self.score_photos=[]
        top=tk.Frame(self.frame,bg="white"); top.pack(fill="x",padx=20,pady=(8,4)); self.dynamic.append(top)
        tk.Label(top,text=f"Manuel scoring – {CONDITIONS[self.condition]['name']}",font=("Arial",22,"bold"),bg="white").pack()
        tk.Label(top,text="Markér de billeder, som deltageren faktisk huskede. Dansk, synonymer og beskrivelser er OK.",font=("Arial",14),bg="white",wraplength=1100).pack(pady=(4,6))
        tk.Label(top,text=f'Deltagerens svar: "{self.raw_response}"',font=("Arial",13),bg="#f2f2f2",wraplength=1100,justify="left",padx=10,pady=7).pack(fill="x",padx=20,pady=(0,6))
        grid=tk.Frame(self.frame,bg="white"); grid.pack(fill="both",expand=True,padx=15); self.dynamic.append(grid)
        for i,p in enumerate(self.current):
            r,c=divmod(i,5); grid.grid_columnconfigure(c,weight=1); grid.grid_rowconfigure(r,weight=1)
            cell=tk.Frame(grid,bg="white",bd=1,relief="solid"); cell.grid(row=r,column=c,padx=4,pady=4,sticky="nsew")
            im=Image.open(p).convert("RGB"); im.thumbnail((150,100)); ph=ImageTk.PhotoImage(im); self.score_photos.append(ph)
            pic=tk.Label(cell,image=ph,bg="white"); pic.pack(pady=(4,2))
            tk.Label(cell,text=label_from_path(p),font=("Arial",10),bg="white",wraplength=160).pack()
            var=tk.BooleanVar(value=False); self.score_vars.append(var)
            tk.Checkbutton(cell,text="Husket",variable=var,font=("Arial",11),bg="white",activebackground="white").pack(pady=(1,4))
            pic.bind("<Button-1>",lambda e,v=var:v.set(not v.get()))
        save=tk.Button(self.frame,text="Gem scoring",font=("Arial",16),command=self.save_scoring); save.pack(pady=8); self.dynamic.append(save)

    def save_scoring(self):
        selected=[1 if v.get() else 0 for v in self.score_vars]
        primacy=sum(selected[:5])/5; middle=sum(selected[5:10])/5; recency=sum(selected[10:15])/5; overall=sum(selected)/15
        pe=primacy-middle; reff=recency-middle
        raw=self.raw_response.replace("\n"," | ")
        with open(self.items_file,"a",newline="",encoding="utf-8") as f:
            w=csv.writer(f)
            for pos,(p,rec) in enumerate(zip(self.current,selected),start=1):
                w.writerow([self.pid,self.session,self.trial,self.condition,self.idx+pos,pos,p.name,label_from_path(p),rec,raw])
        with open(self.trials_file,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([self.pid,self.session,self.trial,self.condition,self.idx+1,self.idx+IMAGES_PER_TRIAL,
                                    sum(selected),overall,primacy,middle,recency,pe,reff,self.math_attempts,self.math_correct,raw])
        self.idx += IMAGES_PER_TRIAL; set_next_index(self.idx)
        self.show_analysis()

    def load_all_trials(self):
        rows=[]
        for fp in RESDIR.glob("*_trials.csv"):
            try:
                with open(fp,newline="",encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        try:
                            cond=row.get("condition","").strip().lower()
                            if cond not in CONDITIONS: continue
                            rows.append({
                                "participant":row.get("participant",""), "condition":cond,
                                "overall":float(row["recall_rate"]), "primacy":float(row["primacy_recall"]),
                                "middle":float(row["middle_recall"]), "recency":float(row["recency_recall"]),
                                "primacy_effect":float(row["primacy_effect"]), "recency_effect":float(row["recency_effect"])
                            })
                        except Exception: pass
            except Exception: pass
        return rows

    def metric_summary(self, rows, key, clamp=False):
        xs=[r[key] for r in rows]
        return mean(xs), ci95(xs,clamp), len(xs)

    def show_analysis(self):
        self.root.unbind("<space>"); self.clear()
        rows=self.load_all_trials()
        if not rows:
            self.main_text.config(text="Ingen gemte trials endnu.")
            b=tk.Button(self.frame,text="Tilbage",font=("Arial",16),command=self.show_menu); b.pack(pady=20); self.dynamic.append(b); return

        outer=tk.Frame(self.frame,bg="white"); outer.pack(fill="both",expand=True,padx=18,pady=10); self.dynamic.append(outer)
        tk.Label(outer,text=f"Samlet statistik – {len(rows)} trials",font=("Arial",22,"bold"),bg="white").pack(pady=(0,8))

        # Per-condition summary
        table=tk.Frame(outer,bg="white"); table.pack()
        headers=["Forsøg","n","Recall","Primacy","Middle","Recency","Primacy effect (95% CI)","Recency effect (95% CI)"]
        widths=[20,4,11,11,11,11,28,28]
        for c,(h,w) in enumerate(zip(headers,widths)):
            tk.Label(table,text=h,font=("Arial",11,"bold"),bg="#e9e9e9",width=w,pady=4).grid(row=0,column=c,padx=1,pady=1)
        summaries={}
        for i,key in enumerate(["standard","fast","arithmetic","pause"],start=1):
            rr=[r for r in rows if r["condition"]==key]
            summaries[key]=rr
            vals=[CONDITIONS[key]["short"],str(len(rr))]
            for metric in ["overall","primacy","middle","recency"]:
                xs=[r[metric] for r in rr]
                vals.append(pct(mean(xs)))
            for metric in ["primacy_effect","recency_effect"]:
                xs=[r[metric] for r in rr]
                lo,hi=ci95(xs,clamp=False)
                ci=ci_text(lo,hi,effect=True)
                vals.append(f"{pp(mean(xs))}   {ci}" if ci != "—" else f"{pp(mean(xs))}   [CI: —]")
            for c,(val,w) in enumerate(zip(vals,widths)):
                tk.Label(table,text=val,font=("Arial",11),bg="white",width=w,pady=4).grid(row=i,column=c,padx=1,pady=1)

        tk.Label(outer,text="Relevante sammenligninger",font=("Arial",17,"bold"),bg="white").pack(pady=(14,5))
        comp=tk.Frame(outer,bg="white"); comp.pack()
        ch=["Sammenligning","Forskel","95% CI"]
        cw=[58,16,26]
        for c,(h,w) in enumerate(zip(ch,cw)):
            tk.Label(comp,text=h,font=("Arial",11,"bold"),bg="#e9e9e9",width=w,pady=4).grid(row=0,column=c,padx=1,pady=1)

        comparisons=[
            ("Hurtig vs. standard: ændring i primacy effect", "fast", "standard", "primacy_effect"),
            ("Plus/minus vs. standard: ændring i recency effect", "arithmetic", "standard", "recency_effect"),
            ("Pause vs. standard: ændring i recency effect", "pause", "standard", "recency_effect"),
            ("Plus/minus vs. pause: ekstra ændring i recency effect", "arithmetic", "pause", "recency_effect"),
        ]
        for i,(label,a,b,metric) in enumerate(comparisons,start=1):
            xa=[r[metric] for r in summaries[a]]; xb=[r[metric] for r in summaries[b]]
            d,lo,hi=diff_ci95(xa,xb)
            vals=[label,pp(d),ci_text(lo,hi,effect=True)]
            for c,(val,w) in enumerate(zip(vals,cw)):
                tk.Label(comp,text=val,font=("Arial",11),bg="white",width=w,pady=4,anchor="w" if c==0 else "center").grid(row=i,column=c,padx=1,pady=1,sticky="nsew")

        tk.Label(outer,text=("Primacy effect = recall(1–5) − recall(6–10).  Recency effect = recall(11–15) − recall(6–10).\n"
                             "95% CI ved hver primacy/recency-effekt er beregnet på trial-niveau. Sammenligningerne viser første betingelse minus anden betingelse."),
                 font=("Arial",11),bg="white",wraplength=1150,justify="center").pack(pady=(8,5))

        # Compact bar chart: primacy and recency effects for all conditions.
        canvas=tk.Canvas(outer,width=1050,height=250,bg="white",highlightthickness=0); canvas.pack(pady=3)
        x0,y0=70,125; chart_h=85
        canvas.create_line(x0,y0,1010,y0,width=2)
        for p in [-100,-50,0,50,100]:
            y=y0-(p/100)*chart_h
            canvas.create_line(x0,y,1010,y,fill="#dddddd")
            canvas.create_text(42,y,text=f"{p:+d} pp",font=("Arial",9))
        x=110
        for cond in ["standard","fast","arithmetic","pause"]:
            rr=summaries[cond]
            for metric,label in [("primacy_effect","P"),("recency_effect","R")]:
                xs=[r[metric] for r in rr]
                v=mean(xs)
                has_value=not math.isnan(v)
                draw_v=v if has_value else 0
                y=y0-draw_v*chart_h
                left=x; right=x+55
                canvas.create_rectangle(left,min(y,y0),right,max(y,y0),fill="#8a8a8a",outline="#555555")
                canvas.create_text((left+right)/2, y-10 if draw_v>=0 else y+10, text=pp(v), font=("Arial",9))
                # 95% CI error bar for the effect, once at least two trials exist.
                lo,hi=ci95(xs,clamp=False)
                if not math.isnan(lo) and not math.isnan(hi):
                    cx=(left+right)/2
                    y_lo=y0-lo*chart_h
                    y_hi=y0-hi*chart_h
                    canvas.create_line(cx,y_hi,cx,y_lo,width=2)
                    canvas.create_line(cx-6,y_hi,cx+6,y_hi,width=2)
                    canvas.create_line(cx-6,y_lo,cx+6,y_lo,width=2)
                canvas.create_text((left+right)/2,225,text=label,font=("Arial",10,"bold"))
                x+=62
            canvas.create_text(x-62,242,text=CONDITIONS[cond]["short"],font=("Arial",10))
            x+=65

        buttons=tk.Frame(outer,bg="white"); buttons.pack(pady=5)
        tk.Button(buttons,text="Tilbage til forsøg",font=("Arial",15),command=self.show_menu).pack(side="left",padx=8)
        tk.Button(buttons,text="Luk program",font=("Arial",15),command=self.quit_program).pack(side="left",padx=8)

    def quit_program(self):
        self.root.destroy()


def main():
    root=tk.Tk()
    try:
        FreeRecallApp(root); root.mainloop()
    except Exception as e:
        messagebox.showerror("Fejl",str(e)); root.destroy()

if __name__=="__main__":
    main()
