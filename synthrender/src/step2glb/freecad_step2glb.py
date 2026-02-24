import os
import sys
import time

def main():
    import FreeCADGui # type: ignore
    import FreeCAD    # type: ignore
    import ImportGui  # type: ignore

    print("Freecad started up!", flush=True)

    mw = FreeCADGui.getMainWindow()
    mw.hide() # Hide GUI


    # New document
    doc_name = "ImportedWithColors"
    doc = FreeCAD.newDocument(doc_name)
    stp_file, output = sys.argv[-2], sys.argv[-1]

    
    # Import via the GUI STEP importer (preserves hierarchy & colors)
    print("Importing step model...", flush=True)

    start = time.time()    
    ImportGui.insert(stp_file, doc.Name)
    doc.recompute()
    print(f"\t-> {time.time()-start:.3f} s", flush=True)
    

    # Tessellate and export exactly as before, including your material flags:
    print("Tesselating model...", flush=True)
    start = time.time()

    for o in doc.Objects:
        if hasattr(o, "Shape"):
            o.Shape.tessellate(1)

    print(f"\t-> {time.time()-start:.3f} s", flush=True)


    # Exporting model with colors:
    print("Exporting model...", flush=True)
    start = time.time()
    objs = [o for o in doc.Objects if not o.InList]
    opts = ImportGui.exportOptions(output)

    if hasattr(opts, "WriteColor"):
        opts.WriteColor = True
    if hasattr(opts, "ConvertColorsToMaterials"):
        opts.ConvertColorsToMaterials = True

    ImportGui.export(objs, output, opts)
    print(f"\t-> {time.time()-start:.3f} s", flush=True)

    print("\nGLB exported →", output)


    # Close document:
    for name in FreeCAD.listDocuments().keys():
        FreeCAD.closeDocument(name)

    mw.close()


if __name__ in ('__main__', os.path.splitext(os.path.basename(__file__))[0]):
    main()