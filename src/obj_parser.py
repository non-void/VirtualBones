"""
My simple obj file parser.
"""

import numpy as np

class Mesh_obj:
    def __init__(self, filename=None):
        self.v = []
        self.vt = []
        self.vn = []
        self.f = []
        self.hasTex = False
        self.hasNorm = False
        self.mtlfile = None
        self.materialList = []
        if filename is not None:
            self.load(filename)

    def load(self, obj_filename):
        obj_file = open(obj_filename, 'r')
        line = obj_file.readline()
        while line:
            if len(line.split()) > 1 and line.split()[0] == 'v':
                self.v.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
            elif len(line.split()) > 1 and line.split()[0] == 'vt':
                self.vt.append([float(line.split()[1]), float(line.split()[2])])
            elif len(line.split()) > 1 and line.split()[0] == 'vn':
                self.vn.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
            elif len(line.split()) > 1 and line.split()[0] == 'f':
                if '//' in line.split()[1] and len(line.split()[1].split('//'))==2:
                    self.hasNorm = True
                    cur_face = []
                    for ver in line.split()[1:]:
                        cur_face.append(int(ver.split('//')[0]))
                    self.f.append(cur_face)
                elif len(line.split()[1].split('/')) ==2:
                    self.hasTex = True
                    cur_face = []
                    for ver in line.split()[1:]:
                        cur_face.append(list(map(int, ver.split('/'))))
                    self.f.append(cur_face)
                elif len(line.split()[1].split('/')) ==3:
                    self.hasTex = True
                    self.hasNorm = True
                    cur_face = []
                    for ver in line.split()[1:]:
                        cur_face.append(ver.split('/'))
                    self.f.append(cur_face)
                else:
                    cur_face = []
                    for ver in line.split()[1:]:
                        cur_face.append(int(ver))
                    self.f.append(cur_face)
            elif 'mtllib ' in line:
                self.mtlfile = line.split()[1]
            line = obj_file.readline()
        obj_file.close()
        # print len(self.v)
        self.v = np.stack(self.v, axis=0)
        if self.f:
            self.f = np.stack(self.f, axis=0)
        if self.vt:
            self.vt = np.stack(self.vt, axis=0)
        if self.vn:
            self.vn = np.stack(self.vn, axis=0)

    def write(self, obj_filename):
        f_out = open(obj_filename, 'w')
        f_out.write('#Export Obj file with Mesh_obj\n')
        #if self.mtlfile != '':
        #    f_out.write('mtllib '+ self.mtlfile + '\n')
        for i in range(self.v.shape[0]):
            f_out.write('v {0} {1} {2}\n'.format(self.v[i,0],self.v[i,1],self.v[i,2]))
        if self.hasTex:
            for i in range(self.vt.shape[0]):
                f_out.write('vt {0} {1}\n'.format(self.vt[i, 0], self.vt[i, 1]))
        if self.hasNorm:
            for i in range(self.vn.shape[0]):
                f_out.write('vn {0} {1} {2}\n'.format(self.vn[i, 0], self.vn[i, 1], self.vn[i, 2]))
        for f in self.f:
            if self.hasTex and self.hasNorm:
                f_out.write('f')
                for i in range(len(f)):
                    f_out.write(' {0}/{1}/{2}'.format(f[i][0],f[i][1],f[i][2]))
                f_out.write('\n')
            elif self.hasTex and not self.hasNorm:
                f_out.write('f')
                for i in range(len(f)):
                    f_out.write(' {0}/{1}'.format(f[i][0], f[i][1]))
                f_out.write('\n')
            elif self.hasNorm and not self.hasTex:
                f_out.write('f')
                for i in range(len(f)):
                    f_out.write(' {0}//{1}'.format(f[i][0], f[i][1]))
                f_out.write('\n')
            elif not self.hasTex and not self.hasNorm:
                f_out.write('f')
                for i in range(len(f)):
                    f_out.write(' {0}'.format(f[i]))
                f_out.write('\n')