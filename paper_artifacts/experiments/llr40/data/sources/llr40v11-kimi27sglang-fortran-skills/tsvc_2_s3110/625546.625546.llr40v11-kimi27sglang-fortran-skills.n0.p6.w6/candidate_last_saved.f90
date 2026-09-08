subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(in) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(inout) :: bb(2, 2)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size

  integer(c_int64_t) :: i, j, tid, nt, t
  real(c_double) :: maxv, chksum, v, mymax
  integer(c_int64_t) :: xindex, yindex, min_idx, myidx
  real(c_double), allocatable :: local_maxv(:)
  integer(c_int64_t), allocatable :: local_idx(:)

  nt = omp_get_max_threads()
  allocate(local_maxv(0:nt - 1))
  allocate(local_idx(0:nt - 1))

  mymax = aa(1, 1)
  myidx = 0

  !$omp parallel private(tid, i, j, v, mymax, myidx)
  tid = omp_get_thread_num()
  mymax = aa(1, 1)
  myidx = 0
  !$omp do schedule(static)
  do j = 1, LEN_2D
    do i = 1, LEN_2D
      v = aa(i, j)
      if (v > mymax) then
        mymax = v
        myidx = (j - 1) * LEN_2D + (i - 1)
      end if
    end do
  end do
  !$omp end do
  local_maxv(tid) = mymax
  local_idx(tid) = myidx
  !$omp end parallel

  maxv = local_maxv(0)
  do t = 1, nt - 1
    if (local_maxv(t) > maxv) maxv = local_maxv(t)
  end do

  min_idx = huge(0_c_int64_t)
  do t = 0, nt - 1
    if (local_maxv(t) == maxv .and. local_idx(t) < min_idx) then
      min_idx = local_idx(t)
    end if
  end do

  xindex = min_idx / LEN_2D
  yindex = mod(min_idx, LEN_2D)

  chksum = maxv + dble(xindex) + dble(yindex)
  bb(1, 1) = chksum
end subroutine tsvc_2_s3110_fp64
