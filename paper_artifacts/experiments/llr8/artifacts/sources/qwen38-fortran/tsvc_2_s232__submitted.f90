subroutine tsvc_2_s232_fp64(aa, bb, n) bind(C, name="tsvc_2_s232_fp64")
  use iso_c_binding
  implicit none
  type(c_ptr), value, intent(in) :: aa, bb
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), contiguous, pointer :: A(:), B(:)
  integer(c_int64_t) :: i, j, base, i0
  real(c_double) :: x, infv, maxf

  if (n < 2) return
  call c_f_pointer(aa, A, [n*n])
  call c_f_pointer(bb, B, [n*n])
  maxf = huge(1.0d0)

  !$omp parallel do schedule(static,1) private(i, base, i0, x, infv)
  do j = 1, n-1
     base = j*n
     x = A(base + 1)
     i0 = j
     do i = 1, j
        x = x*x + B(base + i + 1)
        A(base + i + 1) = x
        if (x > maxf) then
           i0 = i
           infv = x
           exit
        end if
     end do
     if (i0 < j) then
        A(base + i0 + 2 : base + j + 1) = infv
     end if
  end do
  !$omp end parallel do
end subroutine tsvc_2_s232_fp64
