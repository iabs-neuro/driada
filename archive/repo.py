import shutil
import importlib
import git
import sys
import os
from os.path import join


def clone_org_repo(repo, org="iabs-neuro", branch="main", path_to_clone=None):
    rp = git.Repo.clone_from(
        f"https://github.com/{org}/{repo}", branch=branch, to_path=path_to_clone
    )


def reload_module(name, exmodule, absname=None):
    print(f'module "{name}" is already imported, reloading...')
    exmodpath = exmodule.__path__[0]
    shutil.rmtree(exmodpath)
    print(f'Cloning module "{name}" to {exmodpath}...')
    clone_org_repo(name.split(".")[-1], path_to_clone=exmodpath)
    importlib.reload(exmodule)
    print(f'module "{name}" is successfully reloaded')


def import_external_repositories(repolist, extpath="/content", reload_internal=False):
    os.makedirs(extpath, exist_ok=True)

    for repo in repolist:
        rppath = os.path.join(extpath, repo)

        if repo in sys.modules:
            exmodule = sys.modules[repo]
            reload_module(repo, exmodule)

            if reload_internal:
                for module_name in os.listdir(f"{repo}/externals"):
                    intmodname = f"{repo}.externals.{module_name}"
                    intmodule = sys.modules[intmodname]
                    reload_module(intmodname, intmodule)

        else:
            if repo in os.listdir(extpath):
                shutil.rmtree(join(extpath, repo))
            print(f'Cloning module "{repo}" to {extpath}...')
            clone_org_repo(repo, path_to_clone=rppath)
            importlib.import_module(repo, package=None)
