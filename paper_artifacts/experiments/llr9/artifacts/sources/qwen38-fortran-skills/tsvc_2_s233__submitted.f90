subroutine tsvc_2_s233_fp64(aa, bb, cc, len_2d, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(inout) :: bb(len_2d, len_2d)
  real(c_double), intent(in) :: cc(len_2d, len_2d)
  real(c_double), intent(in) :: workspace(*)
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer :: n, nt, t, lo, hi, jlo, jhi, i, j
  real(c_double) :: s

  n = int(len_2d)
  if (n < 9) return

  if (n < 96) then
    ! tiny: plain serial
    do j = 9, n
      do i = 9, n
        aa(i, j) = aa(i, j - 1) + cc(i, j)
      end do
      s = bb(8, j)
      do i = 9, n
        s = s + cc(i, j)
        bb(i, j) = s
      end do
    end do
    return
  end if

  ! Fortran layout (numpy a[j,i] = a[j-1,i] + c[j,i] / b[j,i] = b[j,i-1] + c[j,i]):
  !   A(i,j) = A(i,j-1) + C(i,j)   chain along 2nd index, rows independent
  !   B(i,j) = B(i-1,j) + C(i,j)   chain along 1st index, cols independent
  ! Both reference folds are left folds; reproduced exactly (bit-equal).
  ! No inter-thread data dependence anywhere in this region:
  !   AA: each thread owns a row band across all columns (self-contained chain)
  !   BB: each thread owns a column group (independent per-column folds)

  !$omp parallel
    nt = omp_get_num_threads()
    t = omp_get_thread_num()
    ! AA: row-band stripe; bit-exact per-row left fold, ascending j
    lo = 9 + (n - 8) * t / nt
    hi = 8 + (n - 8) * (t + 1) / nt
    do j = 9, n
      do i = lo, hi
        aa(i, j) = aa(i, j - 1) + cc(i, j)
      end do
    end do
    ! BB: bit-exact per-column left fold; columns ascending so C streams
    jlo = 9 + (n - 8) * t / nt
    jhi = 8 + (n - 8) * (t + 1) / nt
    do j = jlo, jhi
      s = bb(8, j)
      do i = 9, n
        s = s + cc(i, j)
        bb(i, j) = s
      end do
    end do
  !$omp end parallel
end subroutine tsvc_2_s233_fp64
