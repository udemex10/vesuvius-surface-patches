"""Named public scans and fixed physical seeds for the orthogonal surface search."""
import os
CASES={
    'PHerc0125':dict(id='20250821151825',scan='20250821151825-9.362um-1.2m-113keV-masked',pitch=9.362,xc=3400,zc=10467,band=(2700,4380),seeds=[3310,3471,3706,3735]),
    'PHerc0268':dict(id='20251110183117',scan='20251110183117-8.640um-1.2m-116keV-masked',pitch=8.640,xc=6000,zc=7405,band=(1450,2960),seeds=[1820,2193]),
}
SCAN_CASE=os.getenv('FIRST_LETTERS_GRID_CASE','PHerc0125')
CASES['PHerc0268_inner']=dict(CASES['PHerc0268'],specimen='PHerc0268',xc=3800,band=(2430,3880),seeds=[2876,2974,2991])
CASE=CASES[SCAN_CASE]
SPECIMEN=CASE.get('specimen',SCAN_CASE)