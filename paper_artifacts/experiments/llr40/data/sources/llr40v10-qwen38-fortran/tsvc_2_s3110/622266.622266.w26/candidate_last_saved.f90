subroutine tsvc_2_s3110_fp64(aa, bb, len2d) bind(C, name='tsvc_2_s3110_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), intent(in)    :: aa(*)
  real(c_double), intent(inout) :: bb(*)
  integer(c_int64_t), intent(in), value :: len2d

  integer(c_int64_t) :: n2, i, k, k0, xindex, yindex, t
  real(c_double) :: maxv, chksum
  integer(c_int64_t) :: lk(4096)
  logical :: found

  n2 = len2d * len2d
  if (n2 <= 0) return
  maxv = aa(1)
  lk(1:4096) = 0

  !$omp parallel do reduction(max:maxv)
  do i = 1, n2
     if (aa(i) > maxv) maxv = aa(i)
  end do

  !$omp parallel private(k, t, found)
     t = omp_get_thread_num()
     k = 0
     found = .false.
     !$omp do
     do i = 1, n2
        if (.not. found) then
           if (aa(i) == maxv) then
              k = i
              found = .true.
           end if
        end if
     end do
     !$omp end do
     lk(t + 1) = k
  !$omp end parallel

  k = 0
  do t = 1, omp_get_max_threads()
     if (lk(t) > k) k = lk(t)
  end do

  k0 = k - 1
  xindex = k0 / len2d
  yindex = mod(k0, len2d)
  chksum = maxv + dble(xindex) + dble(yindex)
  bb(1) = chksum
end subroutine tsvc_2_s3110_fp64
