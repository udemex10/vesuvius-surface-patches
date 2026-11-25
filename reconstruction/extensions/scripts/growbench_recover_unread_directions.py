"""Run the same outward PHerc. 125 tracker in reflected or rotated charts."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import growbench_recover_unread as route
import growbench_recover_unread_export as export


XC,ZC=3400,10467
READ_CUBE=route.cube


def physical(p,direction,vector=False):
    u,y,v=p
    if vector:
        if direction=="west":return np.array([-u,y,v])
        return np.array([v,y,u if direction=="north" else -u])
    if direction=="west":return np.array([2*XC-u,y,v])
    return np.array([XC+(v-ZC),y,ZC+(u-XC)*(1 if direction=="north" else -1)])


def read_oriented(volume,origin,direction):
    corner=np.minimum(physical(origin,direction),physical(np.array(origin)+256,direction))
    data,keys=READ_CUBE(volume,corner.astype(int))
    if direction=="west":view=data[:,:,::-1]
    elif direction=="north":view=data.transpose(2,1,0)
    else:view=data.transpose(2,1,0)[:,:,::-1]
    return np.ascontiguousarray(view),keys


def native_consensus(directory,direction):
    with np.load(directory/"consensus.npz") as virtual:
        u,v=virtual["x"],virtual["z"]
        if direction=="west":
            x,z=2*XC-u[::-1],v
            transform=lambda grid:grid[:,::-1]
        elif direction=="north":
            x,z=XC+v-ZC,ZC+u-XC
            transform=lambda grid:grid.T
        else:
            x,z=XC+v-ZC,ZC-(u[::-1]-XC)
            transform=lambda grid:grid.T[::-1,:]
        arrays={key:transform(virtual[key]) for key in virtual.files if key not in ("x","z")}
    assert np.all(np.diff(x)>0) and np.all(np.diff(z)>0)
    rz,rx=np.where(arrays["new"])
    if len(rx):
        if direction=="west":
            assert x[rx].max()<=2744 and np.all((z[rz]>=9827)&(z[rz]<=11107))
        else:
            assert np.all((x[rx]>=2760)&(x[rx]<=4040))
            assert z[rz].min()>=11123 if direction=="north" else z[rz].max()<=9811
    np.savez_compressed(directory/"consensus.npz",x=x,z=z,**arrays)
    report=json.loads((directory/"extension.json").read_text())
    for key in ("old_x_max","new_x_min","max_x"):report["outward_chart_"+key]=report.pop(key)
    report.update(direction=direction,physical_coord_system="Original PHerc0125 native XYZ",
                  previous_native_x=[2760,4040],previous_native_z=[9827,11107],
                  approach="native array is reflected or transposed into common outward-growth coordinates per tile")
    (directory/"extension.json").write_text(json.dumps(report,indent=2))
    for path in directory.glob("route_*/*/source.json"):
        data=json.loads(path.read_text())
        data["native_seed_xyz"]=physical(data["site"]["seed_xyz"],direction).tolist()
        data["native_normal_xyz"]=physical(data["site"]["normal_xyz"],direction,True).tolist()
        data["source_frame"]="site and cue array hash use outward chart; keys_zyx index the physical native source"
        path.write_text(json.dumps(data))
    return bool(len(rx))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction",choices=["west","north","south"])
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--export-only",action="store_true")
    args=parser.parse_args()
    direction=args.direction
    directory=route.DEST.parent/("pherc125_"+direction)
    directory.mkdir(exist_ok=True,parents=True)
    export.DIRECTION=direction;export.DEST=directory;export.EXPORT=directory/"flat"
    if not args.export_only:
        with np.load(route.OLD) as old:
            if direction=="west":transform=lambda grid:grid[:,::-1]
            elif direction=="north":transform=lambda grid:grid.T
            else:transform=lambda grid:grid.T[:,::-1]
            np.savez_compressed(directory/"starter_virtual.npz",x=old["x"],z=old["z"],
                                fit=transform(old["fit"]),qualified=transform(old["qualified"]))
        route.OLD=directory/"starter_virtual.npz";route.DEST=directory
        route.cube=lambda volume,origin:read_oriented(volume,origin,direction)
        with ThreadPoolExecutor(args.threads) as pool:list(pool.map(route.route,range(route.NZ)))
        route.combine(projected=False)
        if not native_consensus(directory,direction):return
    folder=export.mesh()
    export.sample(folder)
    export.evidence()


if __name__=="__main__":main()
