	.file	"int16tofloat32.cpp"
	.text
	.section	.text._ZNKSt5ctypeIcE8do_widenEc,"axG",@progbits,_ZNKSt5ctypeIcE8do_widenEc,comdat
	.align 2
	.p2align 4
	.weak	_ZNKSt5ctypeIcE8do_widenEc
	.type	_ZNKSt5ctypeIcE8do_widenEc, @function
_ZNKSt5ctypeIcE8do_widenEc:
.LFB1397:
	.cfi_startproc
	endbr64
	movl	%esi, %eax
	ret
	.cfi_endproc
.LFE1397:
	.size	_ZNKSt5ctypeIcE8do_widenEc, .-_ZNKSt5ctypeIcE8do_widenEc
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC1:
	.string	"+"
.LC2:
	.string	"j"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB1642:
	.cfi_startproc
	endbr64
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	movl	$4096, %esi
	movl	$64, %edi
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	subq	$24, %rsp
	.cfi_def_cfa_offset 64
	movaps	%xmm2, (%rsp)
	call	aligned_alloc@PLT
	movl	$8192, %esi
	movl	$64, %edi
	movq	%rax, %rbx
	call	aligned_alloc@PLT
	movaps	(%rsp), %xmm2
	pxor	%xmm4, %xmm4
	movq	%rax, _ZL10floatArray(%rip)
	movq	%rax, %rdx
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L4:
	movdqa	(%rbx,%rax), %xmm0
	movdqa	%xmm4, %xmm3
	pcmpgtw	%xmm0, %xmm3
	movdqa	%xmm0, %xmm1
	punpcklwd	%xmm3, %xmm1
	punpckhwd	%xmm3, %xmm0
	cvtdq2ps	%xmm1, %xmm1
	cvtdq2ps	%xmm0, %xmm0
	movaps	%xmm1, (%rdx,%rax,2)
	movaps	%xmm0, 16(%rdx,%rax,2)
	addq	$16, %rax
	cmpq	$4096, %rax
	jne	.L4
	movapd	.LC0(%rip), %xmm3
	movq	%rdx, %rax
	leaq	8192(%rdx), %rcx
	.p2align 4,,10
	.p2align 3
.L5:
	movlps	8(%rax), %xmm2
	cvtps2pd	(%rax), %xmm0
	addq	$16, %rax
	mulpd	%xmm3, %xmm0
	cvtps2pd	%xmm2, %xmm1
	mulpd	%xmm3, %xmm1
	cvtpd2ps	%xmm0, %xmm0
	cvtpd2ps	%xmm1, %xmm1
	movlhps	%xmm1, %xmm0
	movaps	%xmm0, -16(%rax)
	cmpq	%rcx, %rax
	jne	.L5
	xorl	%ebx, %ebx
	leaq	_ZNKSt5ctypeIcE8do_widenEc(%rip), %r12
	jmp	.L10
	.p2align 4,,10
	.p2align 3
.L16:
	movsbl	67(%r13), %esi
.L8:
	movq	%rbp, %rdi
	addq	$8, %rbx
	call	_ZNSo3putEc@PLT
	movq	%rax, %rdi
	call	_ZNSo5flushEv@PLT
	cmpq	$8192, %rbx
	je	.L9
	movq	_ZL10floatArray(%rip), %rdx
.L10:
	leaq	_ZSt4cout(%rip), %rdi
	pxor	%xmm0, %xmm0
	cvtss2sd	(%rdx,%rbx), %xmm0
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	movl	$1, %edx
	leaq	.LC1(%rip), %rsi
	movq	%rax, %rdi
	movq	%rax, %rbp
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	_ZL10floatArray(%rip), %rax
	movq	%rbp, %rdi
	pxor	%xmm0, %xmm0
	cvtss2sd	4(%rax,%rbx), %xmm0
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	movl	$1, %edx
	leaq	.LC2(%rip), %rsi
	movq	%rax, %rbp
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	0(%rbp), %rax
	movq	-24(%rax), %rax
	movq	240(%rbp,%rax), %r13
	testq	%r13, %r13
	je	.L15
	cmpb	$0, 56(%r13)
	jne	.L16
	movq	%r13, %rdi
	call	_ZNKSt5ctypeIcE13_M_widen_initEv@PLT
	movq	0(%r13), %rax
	movl	$10, %esi
	movq	48(%rax), %rax
	cmpq	%r12, %rax
	je	.L8
	movq	%r13, %rdi
	call	*%rax
	movsbl	%al, %esi
	jmp	.L8
.L9:
	addq	$24, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	xorl	%eax, %eax
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
.L15:
	.cfi_restore_state
	call	_ZSt16__throw_bad_castv@PLT
	.cfi_endproc
.LFE1642:
	.size	main, .-main
	.p2align 4
	.type	_GLOBAL__sub_I_main, @function
_GLOBAL__sub_I_main:
.LFB2144:
	.cfi_startproc
	endbr64
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	leaq	_ZStL8__ioinit(%rip), %rdi
	call	_ZNSt8ios_base4InitC1Ev@PLT
	movq	_ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
	leaq	__dso_handle(%rip), %rdx
	leaq	_ZStL8__ioinit(%rip), %rsi
	jmp	__cxa_atexit@PLT
	.cfi_endproc
.LFE2144:
	.size	_GLOBAL__sub_I_main, .-_GLOBAL__sub_I_main
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I_main
	.local	_ZL10floatArray
	.comm	_ZL10floatArray,8,8
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC0:
	.long	-1717986918
	.long	1072798105
	.long	-858993459
	.long	1073794252
	.hidden	__dso_handle
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
