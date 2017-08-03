#include <windows.h>
#include <Python.h>
#include <numpy/arrayobject.h>

//#define ULONG unsigned long

#define DEVICEFILENAME "\\\\.\\radacdrv"

#define BUFFERSIZE        1024
#define TXCONFIGSIZE         7
#define RXBUFFERSIZE         2 
#define MESSAGEBUFFERSIZE  256

#define IOCTL_TRANSFER        0x8100E140
#define IOCTL_COLLECT_SAMPLES 0x81006181
#define IOCTL_TX_CONFIG       0x8100E1C0
#define IOCTL_RX_CONFIG       0x8100E200

// Transfer access codes
#define DEVICEWRITE 0
#define DEVICEREAD  1

// Memory map constanst
#define MEMTU  0
#define MEMREG 1
#define MEMTM  2
#define MEMBC  3
#define MEMBC1 4

// Register constants
#define rcControl      0
#define rcSampleDiv    1
#define rcWrap         2
#define rcOutput       3
#define rcNHeaderWords 4
#define rcPulseCount   5
#define rcVerDate      6
#define rcVerNum       7
#define rcFrameCount   8
#define rcIppMask	   9
#define rcIntMask     10
#define rcIntStatus   11
#define rcRfAttn      14

// Control register bit constants
#define crRunTu           0x00000001
#define crRadacIntClkSel  0x00000002
#define crRadacIntTrgSel  0x00000004
#define crRadacSyncSel    0x00000008
#define crSampleEnable    0x00000010
#define crTuSwEnable      0x00000020
#define crEnablePhaseFlip 0x00000080
#define crBeamCodeWrap    0x00000100
#define crEnableDmaIntr   0x00000200
#define crSyncRx          0x10000000
#define crHeaderEnable    0x20000000
#define crSwapIQ          0x40000000
#define crUseRx           0x80000000

// Macros
#define bitOn(bit,value) {((value & bit) == bit)}
	

// Various typedefs

typedef struct
{
	ULONG VersionDate;
	ULONG VersionNumber;
	ULONG RadacHeaderWords;
} RADACINFO;

typedef struct
{
    ULONG Memory;
    ULONG Access;
    ULONG Address;
    ULONG Length;
    ULONG Data[];
} DEVTRANSFER, *pDEVTRANSFER;

typedef struct
{
    BOOL Write;
    ULONG Length;
    ULONG *Data;
} XFERINFO, *pXFERINFO;

typedef struct
{
    BOOL Write;
    ULONG Length;
    ULONG Data[3];
} XFERRX, *pXFERRX;

typedef struct
{
    BOOL Write;
    ULONG Length;
    ULONG Data[7];
} XFERTX, *pXFERTX;

typedef struct
{
	ULONG Length;
} DMATRANSFER;

#ifdef EXPORT
#define ACTION __declspec(dllexport)
#else
#define ACTION __declspec(dllimport)
#endif

ACTION DWORD readRegisters(DWORD mem,DWORD addr,DWORD count,DWORD *outbuf);
ACTION DWORD writeRegisters(DWORD mem,DWORD addr,DWORD count,DWORD *inbuf);
ACTION DWORD readTx(DWORD *buffer);
ACTION DWORD writeTx(DWORD *buffer);
ACTION DWORD readRx(DWORD addr,DWORD *buffer);
ACTION DWORD writeRx(DWORD addr,DWORD *buffer);
ACTION DWORD collectSamples(DWORD count,DWORD *buffer);


