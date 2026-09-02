subroutine tsvc_2_s255_fp64(a, b, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, workspace_size
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i, nt, tid, lo, hi, chunk
  real(c_double) :: p1, p2, cur, c
  c = 0.333d0

  if (LEN_1D < 3_c_int64_t) then
     ! fallback for tiny sizes
     p1 = b(LEN_1D)
     p2 = b(LEN_1D - 1)
     do i = 1, LEN_1D
        cur = b(i)
        a(i) = (cur + p1 + p2) * c
        p2 = p1
        p1 = cur
     end do
     return
  end if

  nt = omp_get_max_threads()
  chunk = (LEN_1D + nt - 1) / nt
  !$omp parallel do private(tid, lo, hi, i, p1, p2, cur)
  do tid = 0, nt - 1
     lo = tid * chunk + 1
     hi = min(lo + chunk - 1, LEN_1D)
     if (lo > hi) cycle
     ! set up the two lagged b-values for this chunk
     if (lo == 1_c_int64_t) then
        p2 = b(LEN_1D - 1)
        p1 = b(LEN_1D)
     else if (lo == 2_c_int64_t) then
        p2 = b(LEN_1D)
        p1 = b(1)
     else
        p2 = b(lo - 2)
        p1 = b(lo - 1)
     end if
     do i = lo, hi
        cur = b(i)
        a(i) = (cur + p1 + p2) * c
        p2 = p1
        p1 = cur
     end do
  end do
end subroutine tsvc_2_s255_fp64
