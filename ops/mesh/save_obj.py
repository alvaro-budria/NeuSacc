import trimesh

import utils as ut
import ops.mesh


def save(filename, V, F, N = None, C = None, T = None, Tidx = None):
    ut.create_parent_directory(filename)
    if ut.get_file_extension(filename) == '.obj':
        if not N:
            N = ops.mesh.compute_normals(V, F)
        _save_obj(filename, V, F, N, C, T=T, Tidx=Tidx)
    else:
        trimesh.Trimesh(vertices=V, faces=F).export(filename)


def _save_obj(filename, V, F, N, C = None, T = None, Tidx = None):

    assert V.size != 0
    assert F.size != 0

    with open(filename, 'w') as f:
        # Write vertices
        for v_id in range(V.shape[0]):
            f.write('v ' + str(V[v_id, 0]) + ' ' +
                    str(V[v_id, 1]) + ' ' +
                    str(V[v_id, 2]))
            if C.size:
                f.write(' ' + str(C[v_id, 0]) + ' ' +
                        str(C[v_id, 1]) + ' ' +
                        str(C[v_id, 2]))
            f.write('\n')

        # Write vertex normals
        for vn_id in range(N.shape[0]):
            f.write('vn ' + str(N[vn_id, 0]) + ' ' +
                    str(N[vn_id, 1]) + ' ' +
                    str(N[vn_id, 2]) + '\n')

        # Write texture coordinates
        if T is not None and T.size:
            for t_id in range(T.shape[0]):
                f.write('vt ' + str(T[t_id, 0]) +
                        ' ' + str(T[t_id, 1]) + '\n')

        # Write faces
        for f_id in range(F.shape[0]):
            if Tidx is None or not Tidx.size:
                f.write('f ' + str(F[f_id, 0] + 1) + ' ' +
                        str(F[f_id, 1] + 1) + ' ' +
                        str(F[f_id, 2] + 1) + '\n')
            else:
                f.write( 'f ' + str( F[ f_id, 0 ] + 1 ) + '/' + str( Tidx[ f_id, 0 ] + 1 ) + \
                            ' ' + str( F[ f_id, 1 ] + 1 ) + '/' + str( Tidx[ f_id, 1 ] + 1 ) + \
                            ' ' + str( F[ f_id, 2 ] + 1 ) + '/' + str( Tidx[ f_id, 2 ] + 1 ) + '\n' )
