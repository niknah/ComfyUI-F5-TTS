import subprocess
import os.path


class Install:
    f5TTSPath = os.path.join(os.path.dirname(__file__), "F5-TTS")

    @staticmethod
    def has_submodule_file():
        return os.path.exists(os.path.join(Install.f5TTSPath, "README.md"))

    @staticmethod
    def check_install():
        if not Install.has_submodule_file():
            Install.install()

    @staticmethod
    def install():
        print("F5TTS. Checking out submodules")
        if not os.path.exists(Install.f5TTSPath):
            os.mkdir(Install.f5TTSPath)
        else:
            try:
                import pygit2
                repo_path = os.path.join(os.path.dirname(__file__))
                repo = pygit2.Repository(repo_path)
                submodules = pygit2.submodules.SubmoduleCollection(repo)
                submodules.update(init=True)
            except Exception as e:
                print(f"pygit2 failed: {e}")
            subprocess.run(
                ['git', 'submodule', 'update', '--init', '--recursive'],
                cwd=os.path.dirname(__file__),
                shell=True,
                )

        if not Install.has_submodule_file():
            f5_tts_repo_url = 'https://github.com/SWivid/F5-TTS'
            subprocess.run(
                ['git', 'clone', '--depth=1', f5_tts_repo_url],
                cwd=os.path.dirname(__file__),
                shell=True,
                )
            path = "F5-TTS"
            if not Install.has_submodule_file():
                try:
                    pygit2.clone_repository(f5_tts_repo_url, path)
                except Exception as e:
                    print(f"pygit2 failed: {e}")

        if not Install.has_submodule_file():
            print("F5TTS. Something is wrong with your git installation.  It is unable to checkout submodules.  You can install the latest from https://git-scm.com/downloads")  # noqa: E501

    @staticmethod
    def clone():
        subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/SWivid/F5-TTS",
                "F5-TTS"
            ],
            cwd=os.path.dirname(__file__)
            )
