subroutine tsvc_2_s3112_fp64(a, b, len_1d) bind(C)
  use, intrinsic :: iso_c_binding
  use, intrinsic :: omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in)    :: a(len_1d)
  real(c_double), intent(out)   :: b(len_1d)
  integer(c_int64_t) :: n, nt64, t64, lo, hi, i
  real(c_double) :: s, offset
  real(c_double), save, allocatable :: part(:)
  integer :: nt, partn

  n = len_1d
  if (n <= 0) return

  nt = 2 * omp_get_max_threads()
  if (nt < 1) nt = 1
  partn = nt + 1
  if (.not. allocated(part)) then
    allocate(part(partn))
  else if (size(part) < partn) then
    deallocate(part)
    allocate(part(partn))
  end if

  nt64 = int(nt, 8)

  !$omp parallel private(t64, lo, hi, i, s, offset) num_threads(nt)
  t64 = omp_get_thread_num()
  lo = (n * t64) / nt64 + 1
  hi = (n * (t64 + 1)) / nt64
  s = 0.0d0
  !$omp simd reduction(+:s)
  do i = lo, hi
    s = s + a(i)
  end do
  part(t64 + 1) = s
  !$omp barrier
  !$omp single
  do i = 2, partn
    part(i) = part(i) + part(i - 1)
  end do
  !$omp end single
  if (t64 < 48) then
    print '(A, I0, 2I0, 2ES15.6)', "P t=", t64, lo, hi, s, part(t64+1)
  end if
  offset = part(t64 + 1) - s
  s = offset
  !$omp simd reduction(inscan, +:s)
  do i = lo, hi
    s = s + a(i)
    !$omp scan inclusive(s)
    b(i) = s
  end do
  !$omp end parallel
end subroutine tsvc_2_s3112_fp64
