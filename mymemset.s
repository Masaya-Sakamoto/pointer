	.file	"mymemset.c"
	.text
	.p2align 4
	.globl	mymemset
	.type	mymemset, @function
mymemset:
.LFB0:
	.cfi_startproc
	endbr64
	testq	%rdx, %rdx
	je	.L6
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	movzbl	%sil, %esi
	call	memset@PLT
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L6:
	movq	%rdi, %rax
	ret
	.cfi_endproc
.LFE0:
	.size	mymemset, .-mymemset
	.ident	"GCC: (Ubuntu 10.5.0-1ubuntu1~22.04) 10.5.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	 1f - 0f
	.long	 4f - 1f
	.long	 5
0:
	.string	 "GNU"
1:
	.align 8
	.long	 0xc0000002
	.long	 3f - 2f
2:
	.long	 0x3
3:
	.align 8
4:
