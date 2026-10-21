/* turbo_inject.c —— 挂起启动 + 注入 DLL（x86 / 32位）
 *
 * 为什么要挂起启动：hook 必须在游戏初始化之前装好。先 CREATE_SUSPENDED
 * 建进程 → 注入 DLL（DllMain 里打 IAT 补丁）→ 再恢复主线程。
 *
 * 用法：
 *   turbo_inject.exe <turbo_hook.dll 路径> <isaac-ng.exe 路径> [游戏参数...]
 *
 * 例（训练实例：20 倍速 + 不渲染）：
 *   set ISAAC_TURBO_SPEED=20
 *   set ISAAC_TURBO_SKIP_PRESENT=1
 *   turbo_inject.exe turbo_hook.dll isaac-ng.exe --luadebug
 *
 * 例（只给你看的那个窗口：正常速度 + 正常渲染）：
 *   set ISAAC_TURBO_SPEED=1
 *   turbo_inject.exe turbo_hook.dll isaac-ng.exe --luadebug
 *
 * 构建：
 *   gcc -m32 -O2 -o turbo_inject.exe turbo_inject.c
 */
#include <windows.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "用法: turbo_inject.exe <dll> <exe> [args...]\n"
                "  环境变量: ISAAC_TURBO_SPEED / ISAAC_TURBO_SKIP_PRESENT / "
                "ISAAC_TURBO_RENDER_EVERY\n");
        return 2;
    }

    const char *dll = argv[1];
    const char *exe = argv[2];

    /* 拼命令行（CreateProcess 要求可写缓冲区） */
    char cmdline[4096];
    cmdline[0] = '\0';
    strncat(cmdline, "\"", sizeof(cmdline) - strlen(cmdline) - 1);
    strncat(cmdline, exe, sizeof(cmdline) - strlen(cmdline) - 1);
    strncat(cmdline, "\"", sizeof(cmdline) - strlen(cmdline) - 1);
    /* argv[2] 就是 exe 本身，游戏参数从 argv[3] 开始 */
    for (int i = 3; i < argc; i++) {
        strncat(cmdline, " ", sizeof(cmdline) - strlen(cmdline) - 1);
        strncat(cmdline, argv[i], sizeof(cmdline) - strlen(cmdline) - 1);
    }

    /* 工作目录必须是游戏目录，否则游戏找不到 resources/ 会直接崩 */
    char workdir[MAX_PATH];
    strncpy(workdir, exe, sizeof(workdir) - 1);
    workdir[sizeof(workdir) - 1] = '\0';
    char *slash = strrchr(workdir, '\\');
    if (!slash)
        slash = strrchr(workdir, '/');
    if (slash)
        *slash = '\0';
    else
        strcpy(workdir, ".");

    char dll_full[MAX_PATH];
    GetFullPathNameA(dll, sizeof(dll_full), dll_full, NULL);

    printf("[inject] exe     = %s\n", exe);
    printf("[inject] dll     = %s\n", dll_full);
    printf("[inject] workdir = %s\n", workdir);
    printf("[inject] cmdline = %s\n", cmdline);
    fflush(stdout);

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    if (!CreateProcessA(NULL, cmdline, NULL, NULL, FALSE,
                        CREATE_SUSPENDED, NULL, workdir, &si, &pi)) {
        fprintf(stderr, "[inject] CreateProcess 失败: %lu\n", GetLastError());
        return 1;
    }
    printf("[inject] 已挂起创建 pid=%lu，开始注入\n", pi.dwProcessId);
    fflush(stdout);

    SIZE_T len = strlen(dll_full) + 1;
    void *remote = VirtualAllocEx(pi.hProcess, NULL, len,
                                  MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!remote) {
        fprintf(stderr, "[inject] VirtualAllocEx 失败: %lu\n", GetLastError());
        TerminateProcess(pi.hProcess, 1);
        return 1;
    }
    if (!WriteProcessMemory(pi.hProcess, remote, dll_full, len, NULL)) {
        fprintf(stderr, "[inject] WriteProcessMemory 失败: %lu\n", GetLastError());
        TerminateProcess(pi.hProcess, 1);
        return 1;
    }

    FARPROC loadlib =
        GetProcAddress(GetModuleHandleA("kernel32.dll"), "LoadLibraryA");
    HANDLE th = CreateRemoteThread(pi.hProcess, NULL, 0,
                                   (LPTHREAD_START_ROUTINE)loadlib,
                                   remote, 0, NULL);
    if (!th) {
        fprintf(stderr, "[inject] CreateRemoteThread 失败: %lu\n", GetLastError());
        TerminateProcess(pi.hProcess, 1);
        return 1;
    }
    WaitForSingleObject(th, 30000);
    DWORD ret = 0;
    GetExitCodeThread(th, &ret);
    CloseHandle(th);

    if (ret == 0) {
        fprintf(stderr, "[inject] LoadLibraryA 返回 0 —— 注入失败"
                        "（位数不匹配？DLL 必须是 32 位）\n");
        TerminateProcess(pi.hProcess, 1);
        return 1;
    }
    printf("[inject] DLL 已加载 (module base=0x%08lX)，恢复主线程\n", ret);
    fflush(stdout);

    ResumeThread(pi.hThread);

    /* 注入器本身退出，游戏继续跑 */
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    printf("[inject] 完成\n");
    return 0;
}
