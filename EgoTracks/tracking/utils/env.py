from iopath.common.file_io import PathManagerFactory

_ENV_SETUP_DONE = False

pathmgr = PathManagerFactory.get(key="tracking")


def setup_environment():
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True
