subroutine tsvc_2_s2233_fp64(aa, bb, cc, len_2d) bind(C, name="tsvc_2_s2233_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  real(c_double), dimension(*), intent(inout) :: aa
  real(c_double), dimension(*), intent(inout) :: bb
  real(c_double), dimension(*), intent(in)    :: cc
  real(c_double), dimension(*), intent(in)    :: len_2d

  integer(c_int64_t) :: n, c0, c0t, r, j, m, ntail
  real(c_double) :: arv(8), brv(8)
  real(c_double) :: ar, br, cv

! len_2d arrives as an int64_t passed by value (C ABI); gfortran passes all
! BIND(C) scalar dummies by reference, so read the value through a double slot
  n = transfer(len_2d(1), 0_c_int64_t)
  if (n <= 9) return

  m = (n - 8_c_int64_t) / 8_c_int64_t
  ntail = mod(n - 8_c_int64_t, 8_c_int64_t)

!$omp parallel do default(none) shared(aa,bb,cc,n,m) private(arv,brv,j,r) schedule(static)
  do c0 = 9_c_int64_t, 8_c_int64_t*m + 1_c_int64_t, 8_c_int64_t
    do j = 1_c_int64_t, 8_c_int64_t
      arv(j) = aa(7_c_int64_t*n + c0 + j - 1_c_int64_t)
      brv(j) = bb(7_c_int64_t*n + c0 + j - 1_c_int64_t)
    end do
    do r = 9_c_int64_t, n
!$omp simd
      do j = 1_c_int64_t, 8_c_int64_t
        arv(j) = arv(j) + cc((r - 1_c_int64_t)*n + c0 + j - 1_c_int64_t)
        aa((r - 1_c_int64_t)*n + c0 + j - 1_c_int64_t) = arv(j)
        brv(j) = brv(j) + cc((r - 1_c_int64_t)*n + c0 + j - 1_c_int64_t)
        bb((r - 1_c_int64_t)*n + c0 + j - 1_c_int64_t) = brv(j)
      end do
!$omp end simd
    end do
  end do
!$omp end parallel do

  if (ntail > 0) then
    c0t = 8_c_int64_t*m + 9_c_int64_t
!$omp parallel do default(none) shared(aa,bb,cc,n,c0t,ntail) private(ar,br,cv,j,r) schedule(static)
    do j = 1_c_int64_t, ntail
      ar = aa(7_c_int64_t*n + c0t + j - 1_c_int64_t)
      br = bb(7_c_int64_t*n + c0t + j - 1_c_int64_t)
      do r = 9_c_int64_t, n
        cv = cc((r - 1_c_int64_t)*n + c0t + j - 1_c_int64_t)
        ar = ar + cv
        aa((r - 1_c_int64_t)*n + c0t + j - 1_c_int64_t) = ar
        br = br + cv
        bb((r - 1_c_int64_t)*n + c0t + j - 1_c_int64_t) = br
      end do
    end do
!$omp end parallel do
  end if
end subroutine tsvc_2_s2233_fp64
