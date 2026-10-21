/* turbo_hook.c —— 以撒的"穷人版虚拟时钟" DLL（x86 / 32位）
 *
 * 思路：不碰游戏二进制，只 hook 它**导入的系统 API**（IAT 打补丁）：
 *   1) QueryPerformanceCounter —— GLFW 的 glfwGetTime 就靠它取时间。
 *      把返回值按倍速放大，游戏看到的"逻辑时间"就跑得比现实快。
 *   2) SwapBuffers / wglSwapBuffers —— 跳过呈现，帧不必等垂直同步、不必上屏。
 *      留 ISAAC_TURBO_RENDER_EVERY 控制每 N 帧放行一次（给"只看一个窗口"用）。
 *
 * 为什么用 IAT 而不是行内 hook：GLFW 是静态链进 exe 的，但它调用
 * QueryPerformanceCounter 仍然走 exe 的导入表，改 IAT 表项就能接管，
 * 不用管指令长度、不用反汇编，风险小得多。
 *
 * 环境变量：
 *   ISAAC_TURBO_SPEED=20        时钟倍速（1 = 关闭）
 *   ISAAC_TURBO_SKIP_PRESENT=1  1 = 跳过 SwapBuffers（0 = 正常呈现）
 *   ISAAC_TURBO_RENDER_EVERY=0  跳过呈现时，每 N 帧放行一次（0 = 一次都不放）
 *   ISAAC_TURBO_LOG=<路径>      写调试日志
 *
 * 构建（32 位，因为游戏是 x86）：
 *   gcc -m32 -shared -O2 -o turbo_hook.dll turbo_hook.c -lkernel32 -luser32 -lgdi32
 */
#include <windows.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

typedef BOOL(WINAPI *QPC_t)(LARGE_INTEGER *);
typedef BOOL(WINAPI *SP_t)(HDC);

static QPC_t real_QPC = NULL;
static SP_t real_SwapBuffers = NULL;

static LARGE_INTEGER qpc_base_real;
static LARGE_INTEGER qpc_last_virtual;
static double g_speed = 1.0;
static int g_skip_present = 0;
static long g_render_every = 0;
static unsigned long g_present_count = 0;
static FILE *g_log = NULL;

static void logf_(const char *fmt, ...)
{
    if (!g_log)
        return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_log, fmt, ap);
    va_end(ap);
    fputc('\n', g_log);
    fflush(g_log);
}

/* ---------------- 虚拟时钟 ---------------- */
static BOOL WINAPI hook_QPC(LARGE_INTEGER *out)
{
    if (!real_QPC(out))
        return FALSE;
    if (g_speed <= 1.0001)
        return TRUE;

    /* 把真实计数按倍速放大：virt = base + (real - base) * speed
     * 单调不减，避免游戏的自检算术炸掉。 */
    LONGLONG delta = out->QuadPart - qpc_base_real.QuadPart;
    LONGLONG virt = qpc_base_real.QuadPart + (LONGLONG)((double)delta * g_speed);
    if (virt < qpc_last_virtual.QuadPart)
        virt = qpc_last_virtual.QuadPart;      /* 保证单调 */
    qpc_last_virtual.QuadPart = virt;
    out->QuadPart = virt;
    return TRUE;
}

/* ---------------- 跳过呈现 ---------------- */
static BOOL WINAPI hook_SwapBuffers(HDC hdc)
{
    if (!g_skip_present)
        return real_SwapBuffers ? real_SwapBuffers(hdc) : TRUE;

    g_present_count++;
    if (g_render_every > 0 && (g_present_count % (unsigned long)g_render_every) == 0) {
        /* 放行一帧，让窗口（和想看训练结果的人）能看到画面 */
        return real_SwapBuffers ? real_SwapBuffers(hdc) : TRUE;
    }
    return TRUE;    /* 不呈现 */
}

/* ---------------- IAT 打补丁 ---------------- */
static void *patch_iat(HMODULE mod, const char *dll_name, const char *func_name, void *new_fn)
{
    BYTE *base = (BYTE *)mod;
    IMAGE_DOS_HEADER *dos = (IMAGE_DOS_HEADER *)base;
    if (dos->e_magic != IMAGE_DOS_SIGNATURE)
        return NULL;
    IMAGE_NT_HEADERS32 *nt = (IMAGE_NT_HEADERS32 *)(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE)
        return NULL;

    IMAGE_DATA_DIRECTORY *dir =
        &nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir->VirtualAddress || !dir->Size)
        return NULL;

    IMAGE_IMPORT_DESCRIPTOR *imp =
        (IMAGE_IMPORT_DESCRIPTOR *)(base + dir->VirtualAddress);
    for (; imp->Name; imp++) {
        const char *dllname = (const char *)(base + imp->Name);
        if (_stricmp(dllname, dll_name) != 0)
            continue;
        IMAGE_THUNK_DATA32 *orig = (IMAGE_THUNK_DATA32 *)(base + imp->OriginalFirstThunk);
        IMAGE_THUNK_DATA32 *thunk = (IMAGE_THUNK_DATA32 *)(base + imp->FirstThunk);
        if (!imp->OriginalFirstThunk || !imp->FirstThunk)
            continue;
        for (; orig->u1.AddressOfData; orig++, thunk++) {
            if (orig->u1.Ordinal & IMAGE_ORDINAL_FLAG32)
                continue;
            IMAGE_IMPORT_BY_NAME *ibn =
                (IMAGE_IMPORT_BY_NAME *)(base + orig->u1.AddressOfData);
            if (strcmp((const char *)ibn->Name, func_name) != 0)
                continue;

            DWORD old_prot;
            void *p = (void *)&thunk->u1.Function;
            if (!VirtualProtect(p, sizeof(void *), PAGE_READWRITE, &old_prot))
                return NULL;
            void *was = (void *)thunk->u1.Function;
            thunk->u1.Function = (DWORD)(DWORD_PTR)new_fn;
            VirtualProtect(p, sizeof(void *), old_prot, &old_prot);
            return was;      /* 返回原函数地址，给转发用 */
        }
    }
    return NULL;
}

static void Init(void)
{
    char buf[512];
    DWORD n;

    n = GetEnvironmentVariableA("ISAAC_TURBO_LOG", buf, sizeof(buf));
    if (n > 0 && n < sizeof(buf))
        g_log = fopen(buf, "a");

    g_speed = 1.0;
    n = GetEnvironmentVariableA("ISAAC_TURBO_SPEED", buf, sizeof(buf));
    if (n > 0 && n < sizeof(buf))
        g_speed = atof(buf);
    if (g_speed < 1.0)
        g_speed = 1.0;

    n = GetEnvironmentVariableA("ISAAC_TURBO_SKIP_PRESENT", buf, sizeof(buf));
    if (n > 0 && n < sizeof(buf))
        g_skip_present = atoi(buf) != 0;

    n = GetEnvironmentVariableA("ISAAC_TURBO_RENDER_EVERY", buf, sizeof(buf));
    if (n > 0 && n < sizeof(buf))
        g_render_every = atol(buf);

    QueryPerformanceCounter(&qpc_base_real);
    qpc_last_virtual = qpc_base_real;

    logf_("[turbo] speed=%.2f skip_present=%d render_every=%ld",
          g_speed, g_skip_present, g_render_every);

    HMODULE exe = GetModuleHandleA(NULL);
    if (g_speed > 1.0001) {
        void *orig = patch_iat(exe, "kernel32.dll", "QueryPerformanceCounter",
                               (void *)hook_QPC);
        if (orig) {
            real_QPC = (QPC_t)orig;
            logf_("[turbo] IAT patched: kernel32!QueryPerformanceCounter -> %p", real_QPC);
        } else {
            logf_("[turbo] 没找到 QueryPerformanceCounter 的 IAT 表项");
        }
    }
    if (g_skip_present) {
        void *orig = patch_iat(exe, "gdi32.dll", "SwapBuffers", (void *)hook_SwapBuffers);
        if (!orig)
            orig = patch_iat(exe, "opengl32.dll", "SwapBuffers", (void *)hook_SwapBuffers);
        if (orig) {
            real_SwapBuffers = (SP_t)orig;
            logf_("[turbo] IAT patched: SwapBuffers -> %p", real_SwapBuffers);
        } else {
            logf_("[turbo] 没找到 SwapBuffers 的 IAT 表项");
        }
    }
}

BOOL WINAPI DllMain(HINSTANCE hinst, DWORD reason, LPVOID reserved)
{
    (void)hinst;
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(hinst);
        Init();
    }
    return TRUE;
}
