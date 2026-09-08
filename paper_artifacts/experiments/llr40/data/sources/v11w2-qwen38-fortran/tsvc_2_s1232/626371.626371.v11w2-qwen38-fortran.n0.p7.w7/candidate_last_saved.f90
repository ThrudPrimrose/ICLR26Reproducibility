subroutine tsvc_2_s1232_fp64(aa, bb, cc, len2d, vlen) bind(C, name="tsvc_2_s1232_fp64")
  use iso_c_binding
  implicit none
  type(c_ptr), value, intent(in) :: aa, bb, cc
  integer(c_int64_t), value, intent(in) :: len2d, vlen
  real(c_double), pointer :: a(:), b(:), c(:)
  integer(c_int64_t) :: n, v, i64, j64, base64, jmax64
  integer :: i, j, jmax, base, n32

  n = len2d
  v = vlen
  if (n <= 0) return
  call c_f_pointer(aa, a, [n*n])
  call c_f_pointer(bb, b, [n*n])
  call c_f_pointer(cc, c, [n*n])

  if (n <= 46340_8) then
    ! Fast path: indices fit in 32 bits -> vectorizable
    n32 = int(n, 4)
    if (v <= 0) then
      !$omp parallel do default(none) private(base,j) shared(a,b,c,n32) schedule(dynamic,16)
      do i = 0, n32-1
        base = i*n32
        !$omp simd
        do j = 1, n32
          a(base+j) = b(base+j) + c(base+j)
        end do
      end do
    else
      !$omp parallel do default(none) private(base,j,jmax,jmax64) shared(a,b,c,n32,n,v) schedule(dynamic,16)
      do i = 0, n32-1
        base = i*n32
        jmax64 = min(int(i, 8), n-1)/v + 1
        jmax = int(jmax64, 4)
        !$omp simd
        do j = 1, jmax
          a(base+j) = b(base+j) + c(base+j)
        end do
      end do
    end if
  else
    ! Large path: 64-bit indices
    if (v <= 0) then
      !$omp parallel do default(none) private(base64,j64) shared(a,b,c,n) schedule(dynamic,16)
      do i64 = 0, n-1
        base64 = i64*n
        do j64 = 1, n
          a(base64+j64) = b(base64+j64) + c(base64+j64)
        end do
      end do
    else
      !$omp parallel do default(none) private(base64,j64,jmax64) shared(a,b,c,n,v) schedule(dynamic,16)
      do i64 = 0, n-1
        base64 = i64*n
        jmax64 = min(i64, n-1)/v + 1
        do j64 = 1, jmax64
          a(base64+j64) = b(base64+j64) + c(base64+j64)
        end do
      end do
    end if
  end if
end subroutine tsvc_2_s1232_fp64
