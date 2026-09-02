! TSVC_2 s3110: global max of a (n,n) matrix + first-occurrence indices.
! numpy: for i in range(n): for j in range(n): if aa[i,j] > maxv: ...
! numpy aa[i,j] == Fortran aa(j+1, i+1): i (outer) is the 2nd Fortran subscript,
! j (inner) the 1st. First occurrence in (i,j) lex order == smallest 2nd
! subscript, then smallest 1st subscript == Fortran column-wise first occurrence.
!
! One memory pass, vectorized:
!  * each OpenMP thread gets a contiguous band of columns (1st subscript
!    innermost: unit stride, SIMD max-reduced per column)
!  * per thread: running max over its band + first column attaining it
!  * serial merge of thread partials (ties -> smaller column index)
!  * x* = first row of the winning column equal to the max (tiny pass)

subroutine tsvc_2_s3110_fp64(aa, bb, len_2d, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d, workspace_size
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(inout) :: bb(2, 2)
  type(c_ptr), value :: workspace
  ! workspace (uint8_t*)/workspace_size unused by this kernel (harmless -Wunused warning)
  integer :: nt, t, y, x, lo, hi, xs, n
  real(c_double) :: cm, tm, mx
  integer :: ty, gy
  real(c_double), allocatable :: mp(:)
  integer, allocatable :: yp(:)
  n = int(len_2d)
  if (n <= 0) return

  nt = omp_get_max_threads()
  if (nt < 1) nt = 1
  if (nt > n) nt = n
  allocate (mp(nt), yp(nt))

  !$omp parallel num_threads(nt) private(cm, tm, ty, y, x, lo, hi, t)
    t = omp_get_thread_num() + 1
    lo = (n * (t - 1)) / nt + 1
    hi = (n * t) / nt
    if (hi >= lo) then
      cm = aa(1, lo)
      !$omp simd reduction(max:cm)
      do x = 2, n
        cm = max(cm, aa(x, lo))
      end do
      tm = cm
      ty = lo
      do y = lo + 1, hi
        cm = aa(1, y)
        !$omp simd reduction(max:cm)
        do x = 2, n
          cm = max(cm, aa(x, y))
        end do
        if (cm > tm) then
          tm = cm
          ty = y
        end if
      end do
      mp(t) = tm
      yp(t) = ty
    else
      mp(t) = -huge(0.0d0)
      yp(t) = n + 1
    end if
  !$omp end parallel

  mx = mp(1)
  gy = yp(1)
  do t = 2, nt
    if (mp(t) > mx .or. (dbit(mp(t)) == dbit(mx) .and. yp(t) < gy)) then
      mx = mp(t)
      gy = yp(t)
    end if
  end do

  xs = findloc(aa(:, gy), mx, dim=1)
  bb(1, 1) = mx + dble(gy - 1) + dble(xs - 1)
contains
  function dbit(a) result(r)
    use iso_c_binding
    implicit none
    real(c_double), intent(in) :: a
    integer(c_int64_t) :: r
    ! bit pattern of a; exact value identity without a real== warning
    r = transfer(a, r)
  end function dbit
end subroutine tsvc_2_s3110_fp64

subroutine s3110_fp64(aa, bb, len_2d, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d, workspace_size
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(inout) :: bb(2, 2)
  type(c_ptr), value :: workspace
  ! workspace (uint8_t*)/workspace_size unused by this kernel (harmless -Wunused warning)
  integer :: nt, t, y, x, lo, hi, xs, n
  real(c_double) :: cm, tm, mx
  integer :: ty, gy
  real(c_double), allocatable :: mp(:)
  integer, allocatable :: yp(:)
  n = int(len_2d)
  if (n <= 0) return

  nt = omp_get_max_threads()
  if (nt < 1) nt = 1
  if (nt > n) nt = n
  allocate (mp(nt), yp(nt))

  !$omp parallel num_threads(nt) private(cm, tm, ty, y, x, lo, hi, t)
    t = omp_get_thread_num() + 1
    lo = (n * (t - 1)) / nt + 1
    hi = (n * t) / nt
    if (hi >= lo) then
      cm = aa(1, lo)
      !$omp simd reduction(max:cm)
      do x = 2, n
        cm = max(cm, aa(x, lo))
      end do
      tm = cm
      ty = lo
      do y = lo + 1, hi
        cm = aa(1, y)
        !$omp simd reduction(max:cm)
        do x = 2, n
          cm = max(cm, aa(x, y))
        end do
        if (cm > tm) then
          tm = cm
          ty = y
        end if
      end do
      mp(t) = tm
      yp(t) = ty
    else
      mp(t) = -huge(0.0d0)
      yp(t) = n + 1
    end if
  !$omp end parallel

  mx = mp(1)
  gy = yp(1)
  do t = 2, nt
    if (mp(t) > mx .or. (dbit(mp(t)) == dbit(mx) .and. yp(t) < gy)) then
      mx = mp(t)
      gy = yp(t)
    end if
  end do

  xs = findloc(aa(:, gy), mx, dim=1)
  bb(1, 1) = mx + dble(gy - 1) + dble(xs - 1)
contains
  function dbit(a) result(r)
    use iso_c_binding
    implicit none
    real(c_double), intent(in) :: a
    integer(c_int64_t) :: r
    ! bit pattern of a; exact value identity without a real== warning
    r = transfer(a, r)
  end function dbit
end subroutine s3110_fp64
