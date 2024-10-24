	.file	"memcpy_asm_empty.c"
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, -108(%rbp)
	movl	$0, -104(%rbp)
	jmp	.L2
.L3:
	movl	-104(%rbp), %eax
	addl	$1, %eax
	imull	-104(%rbp), %eax
	leal	3(%rax), %edx
	movl	-104(%rbp), %eax
	cltq
	movl	%edx, -96(%rbp,%rax,4)
	addl	$1, -104(%rbp)
.L2:
	cmpl	$9, -104(%rbp)
	jle	.L3
	movl	$1, -100(%rbp)
	jmp	.L4
.L5:
	movl	-100(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %edx
	movl	-100(%rbp), %eax
	subl	$1, %eax
	cltq
	movl	-48(%rbp,%rax,4), %eax
	imull	%edx, %eax
	addl	%eax, -108(%rbp)
	addl	$1, -100(%rbp)
.L4:
	cmpl	$9, -100(%rbp)
	jle	.L5
	movl	$0, %eax
	movq	-8(%rbp), %rcx
	xorq	%fs:40, %rcx
	je	.L7
	call	__stack_chk_fail@PLT
.L7:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0"
	.section	.note.GNU-stack,"",@progbits
