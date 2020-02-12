import ndjson
import numpy as np
import tkinter as tk

from datetime import datetime
from esig import tosig as ts

# Modif camille

class Drawing:
    def __init__(self, drawing, do_link_strokes=False, do_rescale=False, link_steps=None, link_timestep=None):
        self.label = drawing.get("word")
        self.countrycode = drawing.get("countrycode")
        self.timestamp = datetime.strptime(drawing.get("timestamp")[:19], "%Y-%m-%d %H:%M:%S")
        self.recognized = drawing.get("recognized")
        self.key_id = drawing.get("key_id")
        self.strokes = [np.transpose(np.array(stroke)) for stroke in drawing.get("drawing")]
        self.nb_strokes = len(drawing.get("drawing"))
        
        if do_link_strokes:
            self.strokes = self.link_strokes(steps=link_steps, timestep=link_timestep)
        
        if do_rescale:
            self.rescale()

    def rescale(self):
        mat = self.concat_drawing()
        min_x = np.min(mat[:, 0])
        min_y = np.min(mat[:, 1])
        max_t = mat[-1, 2]

        for stroke in self.strokes:
            stroke[:, 0] -= min_x
            stroke[:, 1] -= min_y

        mat = self.concat_drawing()
        max_xy = np.max(mat[:, 0:2])

        for stroke in self.strokes:    
            stroke[:, 0:2] = stroke[:, 0:2] / max_xy
            stroke[:, 2] = stroke[:, 2] / max_t

    def link_strokes(self, steps=None, timestep=None):
        res = [np.copy(stroke) for stroke in self.strokes]
        if steps is None and timestep is None:
            timestep = np.mean([(s[-1,2] - s[0,2]) / len(s[:,2]) for s in res])
        for i in reversed(range(self.nb_strokes)):
            res[i] = np.c_[res[i], np.ones(res[i].shape[0])]
            if i == 0:
                break
            if steps is None:
                if not np.isnan(timestep):
                    tmp_steps = max(abs(int((res[i][0,2] - res[i - 1][-1,2]) / timestep)), 2)
                else:
                    tmp_steps = 2
            else:
                tmp_steps = steps
            
            tmp_slice = np.zeros((tmp_steps, 4))
            for dim in range(3):
                tmp_slice[:,dim] = np.linspace(res[i - 1][-1,dim], self.strokes[i][0,dim], num=tmp_steps)
            res.insert(i, tmp_slice)
        
        return res
    
    def concat_drawing(self):
        return np.concatenate(self.strokes, axis=0)
    
    def display(self, scale=100):
        root = tk.Tk()
        canvas = tk.Canvas(
            root,
            height=scale + 10,
            width=scale + 10
        )
        canvas.pack(padx=20, pady=20)
        for stroke in self.strokes:
            if (stroke.shape[1] < 4) or stroke[0,3] == 1:
                fill = "black"
                width=2
            else:
                fill = "red"
                width=1
            for i in range(stroke.shape[0] - 1):
                canvas.create_line(
                    stroke[i,0] * scale + 5,
                    stroke[i, 1] * scale + 5,
                    stroke[i+1, 0] * scale + 5,
                    stroke[i+1, 1] * scale + 5,
                    fill=fill,
                    width=width)
                
        root.mainloop()
    
    def signature(self, degree, log=False):
        return ts.stream2sig(self.concat_drawing(), degree) if not log else ts.stream2logsig(self.concat_drawing(), degree)

if __name__ == '__main__':
    '''
    with open("./data/full_raw_axe.ndjson") as f:
        data = ndjson.load(f)
    
    drawings = [Drawing(draw, do_link_strokes) for draw in data]
    print("ok")
    
    '''
    with open("./data/full_raw_axe.ndjson") as f:
        data = f.readline()
        i=0
        while data:
            if i == 11:
                print("11")
            draw = Drawing(ndjson.loads(data)[0], do_link_strokes=True, do_rescale=True)
            draw.display(scale=300)
            #sig = draw.signature(4)
            #logsig = draw.signature(4, log=True)
            data = f.readline()
            i+=1
    
